/*
 * \file imageio.cc
 */
#include "imageio.hpp"
#include "imageBMP.hpp"

void imagei(CImageBMP &in, unsigned innHpix, unsigned innVpix, char *PixelR, char *PixelG, char *PixelB)
{
    Pixel PixelBUFF;
#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<innVpix;i++)
    {
        for(int j=0;j<innHpix;j++)
        {
            PixelBUFF=in(i,j);
            PixelR[i*innHpix+j]=PixelBUFF.R;
            PixelG[i*innHpix+j]=PixelBUFF.G;
            PixelB[i*innHpix+j]=PixelBUFF.B;
        }
    }
    return ; 
}


void imageo(CImageBMP &out, unsigned innHpix, unsigned innVpix, char *PixelR, char *PixelG, char *PixelB)
{
    Pixel PixelBUFF;
#pragma omp parallel for schedule(dynamic)
    for(unsigned i=0;i<innVpix;i++)
    {
        for(unsigned j=0;j<innHpix;j++)
        {
            PixelBUFF.R=(char)PixelR[i*innHpix+j];
            PixelBUFF.G=(char)PixelG[i*innHpix+j];
            PixelBUFF.B=(char)PixelB[i*innHpix+j];
            out(i,j)=PixelBUFF;
        }
    }
}