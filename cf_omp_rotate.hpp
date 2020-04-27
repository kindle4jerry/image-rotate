/*
 * \file cf_omp_rotate.hpp
 */
#ifndef _CF_OMP_ROTATE_HPP_
#define _CF_OMP_ROTATE_HPP_
#include <omp.h>
#include <cmath>

#include "imageBMP.hpp"
using namespace std;

void c_omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out);
vector<string> arguments(int argc, char* argv[]);

#endif
