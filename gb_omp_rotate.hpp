/*
 * \file gb_omp_rotate.hpp
 */
#ifndef _GB_OMP_ROTATE_HPP_
#define _GB_OMP_ROTATE_HPP_
#include <omp.h>
#include <cmath>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>

#include "imageBMP.hpp"
using namespace std;

__global__
void GetImage();

void c_omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out);
vector<string> arguments(int argc, char* argv[]);

#endif
