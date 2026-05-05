#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <hdf5.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "functions.h"

#define NC_ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(EXIT_FAILURE);}
#define H5_ERR(e) {if ((e) < 0) {printf("HDF5 error in %s:%d\n", __FILE__, __LINE__); exit(EXIT_FAILURE);}}
#define FLOAT_TOLERANCE 1e-5f

static void initNetCDFInputs(NetCDFFile* prec, NetCDFFile* tg, double* time,
                             double* latitudes, double* longitudes) {
  int retval;
  int inLonVarId, inLatVarId, inTimeVarId;

  for (int i = 0; i < NUM_THREADS; i++) {
    if ((retval = nc_open(FILE_NAME_PREC, NC_NOWRITE, &prec[i].ncId))) NC_ERR(retval);
    if ((retval = nc_open(FILE_NAME_TG, NC_NOWRITE, &tg[i].ncId)))     NC_ERR(retval);

    if ((retval = nc_inq_varid(prec[i].ncId, "rr", &prec[i].varId)))   NC_ERR(retval);
    if ((retval = nc_inq_varid(tg[i].ncId, "tg", &tg[i].varId)))       NC_ERR(retval);
  }

  if ((retval = nc_inq_dimid(prec[0].ncId, "longitude", &prec[0].lonId))) NC_ERR(retval);
  if ((retval = nc_inq_dimid(prec[0].ncId, "latitude", &prec[0].latId)))  NC_ERR(retval);
  if ((retval = nc_inq_dimid(prec[0].ncId, "time", &prec[0].timeId)))     NC_ERR(retval);

  if ((retval = nc_inq_varid(prec[0].ncId, "longitude", &inLonVarId))) NC_ERR(retval);
  if ((retval = nc_inq_varid(prec[0].ncId, "latitude", &inLatVarId)))  NC_ERR(retval);
  if ((retval = nc_inq_varid(prec[0].ncId, "time", &inTimeVarId)))     NC_ERR(retval);

  if ((retval = nc_get_var_double(prec[0].ncId, inTimeVarId, time)))      NC_ERR(retval);
  if ((retval = nc_get_var_double(prec[0].ncId, inLonVarId, longitudes))) NC_ERR(retval);
  if ((retval = nc_get_var_double(prec[0].ncId, inLatVarId, latitudes)))  NC_ERR(retval);
}

static void initHDF5Output(HDF5Out* outH5, double* time, const double* latitudes,
                           const double* longitudes, const hsize_t chunk_dims[3]) {
  outH5->file_id = H5Fcreate(FILE_NAME_ANOMALY_H5,
                             H5F_ACC_TRUNC,
                             H5P_DEFAULT,
                             H5P_DEFAULT);
  H5_ERR(outH5->file_id);

  {
    hsize_t dims[3] = { LAT, LON, TIME };
    outH5->dataspace_id = H5Screate_simple(3, dims, NULL);
  }
  H5_ERR(outH5->dataspace_id);

  hid_t plist = H5Pcreate(H5P_DATASET_CREATE);
  H5_ERR(plist);
  H5_ERR(H5Pset_chunk(plist, 3, chunk_dims));
  H5_ERR(H5Pset_fill_time(plist, H5D_FILL_TIME_NEVER));
  H5_ERR(H5Pset_alloc_time(plist, H5D_ALLOC_TIME_EARLY));

  outH5->dataset_id = H5Dcreate(outH5->file_id,
                                "anomaliaClimatica",
                                H5T_STD_I16LE,
                                outH5->dataspace_id,
                                H5P_DEFAULT,
                                plist,
                                H5P_DEFAULT);
  H5_ERR(outH5->dataset_id);
  H5_ERR(H5Pclose(plist));

  {
    hsize_t dim_lat[1] = { LAT };
    hid_t space_lat = H5Screate_simple(1, dim_lat, NULL);
    H5_ERR(space_lat);
    hid_t dset_lat = H5Dcreate(outH5->file_id, "latitude",
                               H5T_IEEE_F64LE, space_lat,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_ERR(dset_lat);
    H5_ERR(H5Dwrite(dset_lat, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, latitudes));
    H5_ERR(H5Dclose(dset_lat));
    H5_ERR(H5Sclose(space_lat));
  }

  {
    hsize_t dim_lon[1] = { LON };
    hid_t space_lon = H5Screate_simple(1, dim_lon, NULL);
    H5_ERR(space_lon);
    hid_t dset_lon = H5Dcreate(outH5->file_id, "longitude",
                               H5T_IEEE_F64LE, space_lon,
                               H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_ERR(dset_lon);
    H5_ERR(H5Dwrite(dset_lon, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, longitudes));
    H5_ERR(H5Dclose(dset_lon));
    H5_ERR(H5Sclose(space_lon));
  }

  {
    hsize_t dim_time[1] = { TIME };
    hid_t space_time = H5Screate_simple(1, dim_time, NULL);
    H5_ERR(space_time);
    hid_t dset_time = H5Dcreate(outH5->file_id, "time",
                                H5T_IEEE_F64LE, space_time,
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5_ERR(dset_time);
    H5_ERR(H5Dwrite(dset_time, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
             H5P_DEFAULT, time));
    H5_ERR(H5Dclose(dset_time));
    H5_ERR(H5Sclose(space_time));
  }

  printf("NetCDF + HDF5 initialization completed.\n");
}

int calculateDroughtCode(PointData *data, double *days) {
    int i;
    int limite = TIME * 0.3;
    int invalid, totalInvalid = 0;
    float prevDroughtCode = 0;
    float efRainfall, moisture, prevMoisture, evapotranspiration;

    data->drought_code[0] = 0.0f;
    for (i = 1; i < TIME; i++) {
        invalid = 0;

        if (data->part_prec_val[i] < 0.0 || data->part_prec_val[i] > 300.0) {
            data->part_prec_val[i] = 0;
            invalid = 1;
        }

        if (data->part_tg_val[i] < -8000 || data->part_tg_val[i] > 8000) {
            data->part_tg_val[i] = 1040;
            invalid = 1;
        }

        if (invalid) totalInvalid++;

        if (data->part_prec_val[i] > 2.8) {
            efRainfall = 0.86 * data->part_prec_val[i] - 1.27;
            prevMoisture = 800 * expf(-prevDroughtCode / 400.0);
            moisture = prevMoisture + 3.937 * efRainfall;
            prevDroughtCode = 400 * log(800.0 / moisture);
            if (prevDroughtCode < 0) prevDroughtCode = 0;
        }

        evapotranspiration = 0.36 * (data->part_tg_val[i] + 2.0) + getDayLength(days[i]);
        if (evapotranspiration < 0) evapotranspiration = 0;
        data->drought_code[i] = prevDroughtCode + 0.5 * evapotranspiration;
        prevDroughtCode = data->drought_code[i];
    }

    return (totalInvalid > limite) ? 1 : 0;
}

int calculateEstresTermico(PointData *data) {
    short percentiles[98];
    int tamano_percentil = TIME / 100;
    for (int i = 1; i < 99; i++)
        percentiles[i - 1] = data->orderedTg[i * tamano_percentil];

    for (int i = 0; i < 7; i++)
        data->estres_termico[i] = -1;

    for (int i = 7; i < TIME; i++) {
        int suma = 0;
        for (int k = 0; k < 8; k++)
            suma += data->part_tg_val[i - k];
        short media = (short)(suma / 8);
        data->estres_termico[i] = encontrarPercentilShort(percentiles, media);
    }
    return 0;
}

int calculateEstresHidrico(PointData *data) {
    float percentiles[98];
    int tamano_percentil = TIME / 100;
    for (int i = 1; i < 99; i++)
        percentiles[i - 1] = data->orderedPrec[i * tamano_percentil];

    for (int i = 0; i < TIME; i++)
        data->estres_hidrico[i] = encontrarPercentilFloat(percentiles, data->drought_code[i]);

    return 0;
}

void calculateAnomaliaClimatica(PointData *data, unsigned long *not_valid) {
    calculateAnomaliaClimatica(data, data->anomalia_climatica, not_valid);
}

void calculateAnomaliaClimatica(PointData *data, short *anomalia, unsigned long *not_valid) {
    for (int i = 0; i < TIME; i++) {
      if (data->estres_termico[i] < 1 || data->estres_termico[i] > 99 ||
          data->estres_hidrico[i] < 1 || data->estres_hidrico[i] > 99) {
          anomalia[i] = -1;
        (*not_valid)++;
      }
      else {
        anomalia[i] = (data->estres_termico[i] + data->estres_hidrico[i]) / 2;
      }
    }
}

void initNetCDFFiles(NetCDFFile *prec, NetCDFFile *tg, NetCDFFile *out, double *time) {
  int retval;
  double latitudes[LAT], longitudes[LON];

  initNetCDFInputs(prec, tg, time, latitudes, longitudes);

  if ((retval = nc_create(FILE_NAME_ANOMALY, NC_NETCDF4, &out->ncId))) NC_ERR(retval);

  if ((retval = nc_def_dim(out->ncId, "latitude", LAT, &out->latId)))  NC_ERR(retval);
  if ((retval = nc_def_dim(out->ncId, "longitude", LON, &out->lonId))) NC_ERR(retval);
  if ((retval = nc_def_dim(out->ncId, "time", TIME, &out->timeId)))    NC_ERR(retval);

  int latVarId, lonVarId, timeVarId;
  if ((retval = nc_def_var(out->ncId, "latitude", NC_DOUBLE, 1, &out->latId, &latVarId))) NC_ERR(retval);
  if ((retval = nc_def_var(out->ncId, "longitude", NC_DOUBLE, 1, &out->lonId, &lonVarId))) NC_ERR(retval);
  if ((retval = nc_def_var(out->ncId, "time", NC_DOUBLE, 1, &out->timeId, &timeVarId))) NC_ERR(retval);

  {
    int dims[3] = {out->latId, out->lonId, out->timeId};
    if ((retval = nc_def_var(out->ncId, "anomaliaClimatica", NC_SHORT, 3, dims, &out->varId))) NC_ERR(retval);
  }

  printf("Using NC_NOFILL\n");
  if ((retval = nc_def_var_fill(out->ncId, out->varId, NC_NOFILL, NULL))) NC_ERR(retval);
  if ((retval = nc_enddef(out->ncId))) NC_ERR(retval);

  if ((retval = nc_put_var_double(out->ncId, latVarId, latitudes)))  NC_ERR(retval);
  if ((retval = nc_put_var_double(out->ncId, lonVarId, longitudes))) NC_ERR(retval);
  if ((retval = nc_put_var_double(out->ncId, timeVarId, time)))      NC_ERR(retval);
}

void initNetCDFandHDF5(NetCDFFile* prec, NetCDFFile* tg, HDF5Out* outH5, double* time) {
  double latitudes[LAT], longitudes[LON];
  const hsize_t chunk_dims[3] = { 1, 1, TIME };

  initNetCDFInputs(prec, tg, time, latitudes, longitudes);
  initHDF5Output(outH5, time, latitudes, longitudes, chunk_dims);
}

void initNetCDFandHDF5ByLat(NetCDFFile* prec, NetCDFFile* tg, HDF5Out* outH5, double* time) {
  double latitudes[LAT], longitudes[LON];
  const hsize_t chunk_dims[3] = { 1, LON, TIME };

  initNetCDFInputs(prec, tg, time, latitudes, longitudes);
  initHDF5Output(outH5, time, latitudes, longitudes, chunk_dims);
}

void closeHDF5Output(HDF5Out *outH5) {
  H5_ERR(H5Fflush(outH5->file_id, H5F_SCOPE_GLOBAL));
  H5_ERR(H5Dclose(outH5->dataset_id));
  H5_ERR(H5Sclose(outH5->dataspace_id));
  H5_ERR(H5Fclose(outH5->file_id));
}

void initWriteQueue(WriteQueue *queue, long capacity) {
  queue->items = (WriteTask **)calloc((size_t)capacity, sizeof(WriteTask *));
  if (queue->items == NULL) {
    printf("Error allocating memory for the write queue\n");
    exit(EXIT_FAILURE);
  }
  queue->capacity = capacity;
  queue->inserted = 0;
  queue->extracted = 0;
}

void destroyWriteQueue(WriteQueue *queue) {
  free(queue->items);
  queue->items = NULL;
  queue->capacity = 0;
  queue->inserted = 0;
  queue->extracted = 0;
}

WriteTask *createWriteTask(size_t values, const size_t start[3], const size_t count[3]) {
  WriteTask *task = (WriteTask *)malloc(sizeof(WriteTask) + sizeof(short) * values);
  if (task == NULL) {
    printf("Error allocating memory for a write task\n");
    exit(EXIT_FAILURE);
  }

  task->anomalia = (short *)(task + 1);

  for (int i = 0; i < 3; i++) {
    task->start[i] = start[i];
    task->count[i] = count[i];
  }

  return task;
}

void destroyWriteTask(WriteTask *task) {
  if (task == NULL) return;
  free(task);
}

void enqueueWriteTask(WriteQueue *queue, WriteTask *task, int threadId, unsigned long invalid) {
  while (1) {
    #pragma omp critical (write_queue)
    {
      if ((queue->inserted - queue->extracted) < queue->capacity) {
        long pos = queue->inserted % queue->capacity;
        queue->items[pos] = task;
        queue->inserted++;

        if (queue->inserted % 5000 == 0) {
          long qsize = queue->inserted - queue->extracted;
          printf("Enqueued: %ld | Queue size: %ld | Thread: %d | Invalid cells: %lu\n",
                 queue->inserted, qsize, threadId, invalid);
        }

        task = NULL;
      }
    }

    if (task == NULL) break;
    usleep(10);
  }
}

WriteTask *dequeueWriteTask(WriteQueue *queue) {
  WriteTask *task = NULL;

  #pragma omp critical (write_queue)
  {
    if (queue->extracted < queue->inserted) {
      long pos = queue->extracted % queue->capacity;
      task = queue->items[pos];
      queue->items[pos] = NULL;
      queue->extracted++;

      if (queue->extracted % 5000 == 0) {
        long qsize = queue->inserted - queue->extracted;
        printf("Dequeued: %ld | Queue size: %ld\n", queue->extracted, qsize);
      }
    }
  }

  return task;
}

int shortCompare(const void *p1, const void *p2){
  short short_a = *((short *)p1);
  short short_b = *((short *)p2);
  if(short_a == short_b)
    return 0;
  else if(short_a < short_b)
    return -1;
  else
    return 1;
}

int floatCompare(const void *p1, const void *p2){
  float float_a = *((float *)p1);
  float float_b = *((float *)p2);
  if(float_a == float_b)
    return 0;
  else if(float_a < float_b)
    return -1;
  else
    return 1;
}

double getDayLength(double day){
  int dayYear = (int) day%365;
  if(dayYear < 90)
    return -1.6;
  if(dayYear<120)
    return 0.9;
  if(dayYear<151)
    return 3.8;
  if(dayYear<181)
    return 5.8;
  if(dayYear<212)
    return 6.4;
  if(dayYear<243)
    return 5.0;
  if(dayYear<273)
    return 2.4;
  if(dayYear<304)
    return 0.4;
  return -1.6;
}

int encontrarPercentilShort(short *percentiles, short valor) {
    int left = 0, right = 97, result = 98;
    while (left <= right) {
        int mid = (left + right) / 2;
        if (valor <= percentiles[mid]) {
            result = mid + 1;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}

int encontrarPercentilFloat(float *percentiles, float valor) {
    int left = 0, right = 97, result = 98;
    while (left <= right) {
        int mid = (left + right) / 2;
        float diff = valor - percentiles[mid];
        if (diff < -FLOAT_TOLERANCE) {
            result = mid + 1;
            right = mid - 1;
        } else if (fabsf(diff) < FLOAT_TOLERANCE) {
            return mid + 1;
        } else {
            left = mid + 1;
        }
    }
    return result;
}
