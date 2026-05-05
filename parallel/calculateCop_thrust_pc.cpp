#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <omp.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include "case_study.h"
#include "functions.h"

#include <thrust/sort.h>
#include <thrust/execution_policy.h>

int NUM_THREADS = 1;
#define NC_ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(EXIT_FAILURE);}
#define GLOBAL_QUEUE_SIZE 10000

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

    WriteQueue writeQueue;
    initWriteQueue(&writeQueue, GLOBAL_QUEUE_SIZE);

    unsigned long notCalculated = 0;
    unsigned long total_not_valid = 0;
    double read_time = 0.0;
    double write_time = 0.0;

    double start_time = omp_get_wtime();

    #pragma omp parallel shared(time, writeQueue, outFile, write_time) reduction(+:total_not_valid, notCalculated, read_time) num_threads(NUM_THREADS + 1)
    {
      PointData data;
      int i, threadId = omp_get_thread_num();
      int retval;
      unsigned long local_not_valid = 0;
      unsigned long tasks_completed = 0;

      if (threadId == NUM_THREADS) {
        printf("Consumer thread %d active...\n", threadId);
        double start_1 = omp_get_wtime();
        double start_10000 = omp_get_wtime();
        int tasksWritten = 0;
        const int totalTasks = LAT * LON;

        while (tasksWritten < totalTasks) {
          WriteTask *task = dequeueWriteTask(&writeQueue);
          if (task) {
            double write_start = omp_get_wtime();
            if ((retval = nc_put_vara_short(outFile.ncId, outFile.varId,
                                            task->start, task->count,
                                            task->anomalia))) NC_ERR(retval);
            double write_end = omp_get_wtime();
            write_time += write_end - write_start;
            destroyWriteTask(task);
            tasksWritten++;
          } else {
            usleep(10);
          }

          if (tasksWritten == 1) {
            double end_1 = omp_get_wtime();
            printf("Time to write 1 task: %f seconds\n", end_1 - start_1);
          }
          if (tasksWritten == 10000) {
            double end_10000 = omp_get_wtime();
            printf("Time to write %d tasks: %f seconds\n", tasksWritten, end_10000 - start_10000);
          }
        }

        printf("Consumer finished with %d tasks\n", tasksWritten);
      }
      else {

        double read_start, read_end;
        #pragma omp for collapse(2) schedule(guided) nowait
        for (int j = 0; j < LAT; j++) {
          for (int k = 0; k < LON; k++) {
            size_t start[3] = {(size_t)j, (size_t)k, 0};
            size_t count[3] = {1, 1, TIME};
            WriteTask *task = createWriteTask(TIME, start, count);
            short *anomalia = task->anomalia;

            read_start = omp_get_wtime();
            if ((retval = nc_get_vara_short(tgFiles[threadId].ncId, tgFiles[threadId].varId, start, count, data.part_tg_val))) NC_ERR(retval);
            if ((retval = nc_get_vara_float(precFiles[threadId].ncId, precFiles[threadId].varId, start, count, data.part_prec_val))) NC_ERR(retval);
            read_time += omp_get_wtime() - read_start;

            int invalidValue = calculateDroughtCode(&data, time);
            if (!invalidValue) {
              memcpy(data.orderedTg, data.part_tg_val, sizeof(short) * TIME);
              thrust::sort(thrust::host, data.orderedTg, data.orderedTg + TIME);
              calculateEstresTermico(&data);

              memcpy(data.orderedPrec, data.drought_code, sizeof(float) * TIME);
              thrust::sort(thrust::host, data.orderedPrec, data.orderedPrec + TIME);
              calculateEstresHidrico(&data);

              calculateAnomaliaClimatica(&data, anomalia, &local_not_valid);
            } else {
              notCalculated++;
              for (i = 0; i < TIME; i++) {
                anomalia[i] = -1;
                local_not_valid++;
              }
            }

            tasks_completed++;
            enqueueWriteTask(&writeQueue, task, threadId, notCalculated);
          }
        }

        total_not_valid += local_not_valid;
        printf("Producer thread: %d finished with %lu tasks , Invalid cells: %lu , Invalid values: %lu\n",
               threadId, tasks_completed, notCalculated, local_not_valid);
      }
    }

    double end_time = omp_get_wtime();
    double seconds = end_time - start_time;
    printf("Computation completed. NotCalculated: %ld, Total invalid values: %ld\n", notCalculated, total_not_valid);
    printf("Total read time: %f seconds\n", read_time);
    printf("Total write time: %f seconds\n", write_time);
    printf("Total duration: %f seconds\n\n", seconds);

    for (int i = 0; i < NUM_THREADS; i++) {
      if ((retval = nc_close(precFiles[i].ncId))) NC_ERR(retval);
      if ((retval = nc_close(tgFiles[i].ncId))) NC_ERR(retval);
    }
    if ((retval = nc_close(outFile.ncId))) NC_ERR(retval);
    destroyWriteQueue(&writeQueue);

    return 0;
}
