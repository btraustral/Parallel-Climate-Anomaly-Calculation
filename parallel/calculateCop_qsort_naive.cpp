#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <omp.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "case_study.h"
#include "functions.h"

int NUM_THREADS = 1;  
#define NC_ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(EXIT_FAILURE);}

// -----------------------------
// MAIN
// -----------------------------
int main(int argc, char *argv[]) {

    if (argc != 2) {
      printf("Usage: %s <num_threads>\n", argv[0]);
      printf("Max threads: %d\n", omp_get_max_threads());
      return EXIT_FAILURE;
    }
    NUM_THREADS = atoi(argv[1]);
    NetCDFFile precFiles[NUM_THREADS], tgFiles[NUM_THREADS], outFile;
    int retval;
    double time[TIME];
    initNetCDFFiles(precFiles, tgFiles, &outFile, time);
    unsigned long notCalculated=0; // Total invalid cells
    unsigned long total_not_valid = 0; // Total invalid values

    double read_time = 0.0;
    double calc_time = 0.0;
    double write_time = 0.0;
    double effective_write_time = 0.0;
    double start_time, end_time;

    size_t count[3] = {1, 1, TIME};

    start_time = omp_get_wtime();

    #pragma omp parallel shared(time) reduction(+:total_not_valid, notCalculated, read_time, calc_time, write_time, effective_write_time) num_threads(NUM_THREADS)
    {
      PointData data;
      int i, threadId = omp_get_thread_num();
      int retval;
      unsigned long local_not_valid = 0;

      #pragma omp for collapse(2) schedule(guided)
      for (int j = 0; j < LAT; j++) {
        for (int k = 0; k < LON; k++) {
          size_t start[3] = {(size_t)j, (size_t)k, 0};

          double t_read_start, t_read_end;
          double t_calc_start, t_calc_end;
          double t_write_start, t_write_end;
          double t_effective_write_start, t_effective_write_end;

          t_read_start = omp_get_wtime();
          if ((retval = nc_get_vara_short(tgFiles[threadId].ncId, tgFiles[threadId].varId, start, count, data.part_tg_val))) NC_ERR(retval);
          if ((retval = nc_get_vara_float(precFiles[threadId].ncId, precFiles[threadId].varId, start, count, data.part_prec_val))) NC_ERR(retval);
          t_read_end = omp_get_wtime();
          read_time += t_read_end - t_read_start;

          t_calc_start = omp_get_wtime();
          int invalidValue = calculateDroughtCode(&data, time);
          if(!invalidValue)
          {
            memcpy(data.orderedTg, data.part_tg_val, sizeof(short) * TIME);
            qsort(data.orderedTg, TIME, sizeof(short), shortCompare);
            calculateEstresTermico(&data);
    
            memcpy(data.orderedPrec, data.drought_code, sizeof(float) * TIME);
            qsort(data.orderedPrec, TIME, sizeof(float), floatCompare);
            calculateEstresHidrico(&data);
    
            calculateAnomaliaClimatica(&data, &local_not_valid);
          }
          else
          {
            notCalculated++;
            for(i=0; i<TIME; i++){
              data.anomalia_climatica[i]=-1;
              local_not_valid++;
            }
          }
          t_calc_end = omp_get_wtime();
          calc_time += t_calc_end - t_calc_start;

          t_write_start = omp_get_wtime();
          #pragma omp critical
          {
            t_effective_write_start = omp_get_wtime();
            if ((retval = nc_put_vara_short(outFile.ncId, outFile.varId, start, count, data.anomalia_climatica))) NC_ERR(retval);
            t_effective_write_end = omp_get_wtime();
            effective_write_time += t_effective_write_end - t_effective_write_start;
          } 
          t_write_end = omp_get_wtime();
          write_time += t_write_end - t_write_start;
        } // END LONGITUDE LOOP
      } // END LATITUDE LOOP
      total_not_valid += local_not_valid;
      printf("Thread: %d finished, Invalid cells: %lu , Invalid values: %lu\n", threadId, notCalculated, local_not_valid);
    } // END PRAGMA
    
    end_time = omp_get_wtime();
    double seconds = end_time - start_time;
    printf("Computation completed. NotCalculated: %ld, Total invalid values: %ld\n",notCalculated, total_not_valid);
    printf("Total duration: %f seconds\n\n", seconds);

    printf("Total read time: %f seconds\n", read_time);
    printf("Total calculation time: %f seconds\n", calc_time);
    printf("Total write time: %f seconds\n", write_time);
    printf("Effective write time: %f seconds\n", effective_write_time);

    for (int i = 0; i < NUM_THREADS; i++) {
      if ((retval = nc_close(precFiles[i].ncId))) NC_ERR(retval);
      if ((retval = nc_close(tgFiles[i].ncId))) NC_ERR(retval);
    }
    if ((retval = nc_close(outFile.ncId))) NC_ERR(retval);

    return 0;
}
