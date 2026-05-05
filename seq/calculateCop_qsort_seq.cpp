#include <stdio.h>
#include <stdlib.h>
#include <netcdf.h>
#include <omp.h>
#include <math.h>
#include <string.h>
#include "case_study.h" 
#include "functions.h"

int NUM_THREADS = 1;
#define NC_ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(EXIT_FAILURE);}

// -----------------------------
// MAIN
// -----------------------------
int main() {
    NetCDFFile precFile, tgFile, outFile;
    int retval;
    double time[TIME];
    initNetCDFFiles(&precFile, &tgFile, &outFile, time);
    unsigned long notCalculated = 0;
    unsigned long total_not_valid = 0;

    double t_start, t_end; // TOTAL
    double t_start_read, t_end_read; // READ
    double t_start_calc, t_end_calc; // CALCULATION
    double t_start_write, t_end_write; // WRITE

    double calc_time = 0.0;
    double write_time = 0.0;
    double read_time = 0.0;

    size_t count[3] = {1, 1, TIME};

    t_start = omp_get_wtime(); // START TOTAL TIMER
    PointData data;
    for (int j = 0; j < LAT; j++) {
      for (int k = 0; k < LON; k++) {
        size_t start[3] = {(size_t)j, (size_t)k, 0};

        t_start_read = omp_get_wtime(); // START READ TIMER
        if ((retval = nc_get_vara_short(tgFile.ncId, tgFile.varId, start, count, data.part_tg_val))) NC_ERR(retval);
        if ((retval = nc_get_vara_float(precFile.ncId, precFile.varId, start, count, data.part_prec_val))) NC_ERR(retval);
        t_end_read = omp_get_wtime(); // END READ TIMER
        read_time += t_end_read - t_start_read;

        t_start_calc = omp_get_wtime(); // START CALCULATION TIMER (EXCLUDING READ)
        int invalidValue = calculateDroughtCode(&data, time);
        if (!invalidValue) {
            memcpy(data.orderedTg, data.part_tg_val, sizeof(short) * TIME);
            qsort(data.orderedTg, TIME, sizeof(short), shortCompare);
            calculateEstresTermico(&data);

            memcpy(data.orderedPrec, data.drought_code, sizeof(float) * TIME);
            qsort(data.orderedPrec, TIME, sizeof(float), floatCompare);
            calculateEstresHidrico(&data);

            calculateAnomaliaClimatica(&data, &total_not_valid);
        } 
        else {
          notCalculated++;
          for (int i = 0; i < TIME; i++) {
            data.estres_termico[i] = -1;
            data.estres_hidrico[i] = -1;
            data.anomalia_climatica[i] = -1;
            total_not_valid++;
          }
        }
        t_end_calc = omp_get_wtime(); // END CALCULATION TIMER
        calc_time += t_end_calc - t_start_calc;
        
        t_start_write = omp_get_wtime(); // START WRITE TIMER
        if ((retval = nc_put_vara_short(outFile.ncId, outFile.varId, start, count, data.anomalia_climatica))) NC_ERR(retval);
        t_end_write = omp_get_wtime(); // END WRITE TIMER
        write_time += t_end_write - t_start_write;
      }
    }
    t_end = omp_get_wtime();
    double seconds = t_end - t_start;
    printf("Computation completed. NotCalculated: %ld, Total invalid values: %ld\n", notCalculated, total_not_valid);
    printf("Total read time: %f seconds\n", read_time);
    printf("Total calculation time: %f seconds\n", calc_time);
    printf("Total write time: %f seconds\n", write_time);
    printf("Total duration: %f seconds\n\n", seconds);

    if ((retval = nc_close(precFile.ncId))) NC_ERR(retval);
    if ((retval = nc_close(tgFile.ncId))) NC_ERR(retval);
    if ((retval = nc_close(outFile.ncId))) NC_ERR(retval);
    
    return 0;
}
