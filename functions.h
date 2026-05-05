#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <stddef.h>
#include <hdf5.h>
#include "case_study.h"
extern int NUM_THREADS;

// -----------------------------
// Data structures
// -----------------------------
typedef struct {
  short part_tg_val[TIME];
  float part_prec_val[TIME];
  float drought_code[TIME];
  short estres_termico[TIME];
  short estres_hidrico[TIME];
  short anomalia_climatica[TIME];
  short orderedTg[TIME];
  float orderedPrec[TIME];
} PointData;

typedef struct {
  int ncId;
  int varId;
  int lonId;
  int latId;
  int timeId;
} NetCDFFile;

typedef struct {
  hid_t file_id;
  hid_t dataset_id;
  hid_t dataspace_id;
} HDF5Out;

typedef struct {
  short *anomalia;
  size_t start[3];
  size_t count[3];
} WriteTask;

typedef struct {
  WriteTask **items;
  long capacity;
  volatile long inserted;
  volatile long extracted;
} WriteQueue;

// -----------------------------
// Helper functions
// -----------------------------
int calculateDroughtCode(PointData *data, double *days);
int encontrarPercentilShort(short *percentiles, short valor);
int encontrarPercentilFloat(float *percentiles, float valor);
int calculateEstresTermico(PointData *data);
int calculateEstresHidrico(PointData *data);
void calculateAnomaliaClimatica(PointData *data, unsigned long *not_valid);
void calculateAnomaliaClimatica(PointData *data, short *anomalia, unsigned long *not_valid);
void initNetCDFFiles(NetCDFFile *prec, NetCDFFile *tg, NetCDFFile *out, double *time);
void initNetCDFandHDF5(NetCDFFile* prec, NetCDFFile* tg, HDF5Out* outH5, double* time);
void initNetCDFandHDF5ByLat(NetCDFFile* prec, NetCDFFile* tg, HDF5Out* outH5, double* time);
void closeHDF5Output(HDF5Out *outH5);
void initWriteQueue(WriteQueue *queue, long capacity);
void destroyWriteQueue(WriteQueue *queue);
WriteTask *createWriteTask(size_t values, const size_t start[3], const size_t count[3]);
void destroyWriteTask(WriteTask *task);
void enqueueWriteTask(WriteQueue *queue, WriteTask *task, int threadId, unsigned long invalid);
WriteTask *dequeueWriteTask(WriteQueue *queue);
int shortCompare(const void *p1, const void *p2);
int floatCompare(const void *p1, const void *p2);
double getDayLength(double day);


#endif
