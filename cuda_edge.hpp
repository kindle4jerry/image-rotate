/*
 * \file cuda_edge.hpp
 */
#ifndef _CUDA_EDGE_HPP_
#define _CUDA_EDGE_HPP_

#include "imageBMPcuda.hpp"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <iomanip>

//#define	CEIL(a,b)				((a+b-1)/b)

double const INMB = 1.0/1024/1024;
double const INGB = 1.0/1024/1024/1024;

// Ceiling operation 
inline unsigned host_ceil(unsigned const a, unsigned const b) {return (a+b-1)/b; }
// convert size in Bytes to MB
inline float set_size_MB(size_t const size ){return (float) size*INMB; }
// compute the data transfer bandwidth in GB
inline float set_BW_GB(size_t const size, float const time ){return (float) size*INGB/time; }

double const PI = 3.1415926;
unsigned char const  ETRUE	= 0;
unsigned char const  EFALSE= 255;

int launch_edge_kernel(CImageBMP &image, unsigned const ThreshLo, unsigned const ThreshHi, unsigned const nThdsPerBlk);

__global__
void bw_kernel(double *bwMat, unsigned char *gpuMat, unsigned nHpix);

__global__
void gauss_kernel(double *gaussMat, double *bwMat, unsigned nHpix, unsigned nVpix);


__global__
void sobel_kernel(double *gradMat, double *thetaMat, double *gaussMat, unsigned nHpix, unsigned nVpix);

__global__
void prune_kernel(unsigned char *edgeImg, double *gradMat, double *thetaMat, unsigned nHpix, unsigned nVpix, unsigned ThreshLo, unsigned ThreshHi);

#endif
