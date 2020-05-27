/*
 * \file imageio.hpp
 */
#ifndef _IMAGEIO_HPP_
#define _IMAGEIO_HPP_
#include <omp.h>
#include <cmath>
#include "imageBMP.hpp"
using namespace std;

void imagei(CImageBMP &in, unsigned innHpix, unsigned innVpix, char *PixelR, char *PixelG, char *PixelB);
void imageo(CImageBMP &out, unsigned innHpix, unsigned innVpix, char *PixelR, char *PixelG, char *PixelB);

#endif
