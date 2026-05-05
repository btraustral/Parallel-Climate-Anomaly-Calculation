#ifndef CASE_STUDY_H
#define CASE_STUDY_H

#ifdef SMALL
    #define FILE_NAME_PREC "./cop_prec_small.nc"
    #define FILE_NAME_TG   "./cop_tg_small.nc"
    #define FILE_NAME_ANOMALY "cop_anomaly_small.nc"
    #define FILE_NAME_ANOMALY_H5 "cop_anomaly_small.h5"
    #define LAT 166
    #define LON 222
    #define TIME 3287
#elif defined(MEDIUM)
    #define FILE_NAME_PREC "./cop_prec_medium.nc"
    #define FILE_NAME_TG   "./cop_tg_medium.nc"
    #define FILE_NAME_ANOMALY "cop_anomaly_medium.nc"
    #define FILE_NAME_ANOMALY_H5 "cop_anomaly_medium.h5"
    #define LAT 437
    #define LON 592
    #define TIME 5844
#elif defined(BIG)
    #define FILE_NAME_PREC "./cop_prec_big.nc"
    #define FILE_NAME_TG   "./cop_tg_big.nc"
    #define FILE_NAME_ANOMALY "cop_anomaly_big.nc"
    #define FILE_NAME_ANOMALY_H5 "cop_anomaly_big.h5"
    #define LAT 434
    #define LON 575
    #define TIME 27028
#else
    #error "Define SMALL, MEDIUM, or BIG at compile time"
#endif

#endif 
