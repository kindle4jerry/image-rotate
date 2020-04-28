/*
 * \file b_omp_rotate.hpp
 */
#ifndef _B_OMP_ROTATE_HPP_
#define _B_OMP_ROTATE_HPP_
#include <omp.h>
#include <cmath>

#include "imageBMP.hpp"
using namespace std;

void c_omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out);
vector<string> arguments(int argc, char* argv[]);

//Define a double pixel but not weight for backward rotate
typedef struct PixelDouble{
        double R;
        double G;
        double B;

} CPixelDouble;

#endif
