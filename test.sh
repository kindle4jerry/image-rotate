#!/bin/sh
#ENVIRONMENT
INPUT_BMP=tank

#COMMAND
echo "No anti-dogtooth"
./xc_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-c.bmp 45
./xc_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-c.bmp 45
./xc_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-c.bmp 45
echo "Forward with anti-dogtooth"
./xcf_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45
./xcf_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45
./xcf_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45
echo "Backward with anti-dogtooth"
./xcb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45
./xcb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45
./xcb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45
echo "OpenMP"
for((i=1;i<=4;i++));  
do   
./xcb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-omp.bmp 45 $i
./xcb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-omp.bmp 45 $i
./xcb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-omp.bmp 45 $i
done
echo "CUDA"
./xgb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-gb.bmp 45
./xgb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-gb.bmp 45
./xgb_omp_rotate.x $INPUT_BMP.bmp $INPUT_BMP-45-gb.bmp 45
