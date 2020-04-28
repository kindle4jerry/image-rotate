#!/bin/sh
#ENVIRONMENT
INPUT_BMP=tank

#COMMAND
echo "No anti-dogtooth"
./xc_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-c.bmp 45
./xc_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-c.bmp 45
./xc_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-c.bmp 45
echo "Forward with anti-dogtooth"
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45
echo "Backward with anti-dogtooth"
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45
<<COMMENT
echo "Forward with anti-dogtooth"
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 2
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 2
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 2
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 4
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 4
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 4
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 8
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 8
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 8
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 16
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 16
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 16
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 32
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 32
./xcf_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cf.bmp 45 32
COMMENT
echo "Backward with anti-dogtooth"
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 2
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 2
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 2
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 4
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 4
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 4
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 8
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 8
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 8
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 16
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 16
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 16
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 32
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 32
./xcb_omp_rorate.x $INPUT_BMP.bmp $INPUT_BMP-45-cb.bmp 45 32
