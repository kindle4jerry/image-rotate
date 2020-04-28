/*
 * \file b_omp_rotate.cc
 */
#include "b_omp_rotate.hpp"

void c_omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out)
{
    // Initial a matrix for save the input pixel
    Pixel **PixelMatrixBUFF;
    PixelMatrixBUFF=new Pixel*[in.nVpix];
    for(int i=0;i<in.nVpix;i++)
    {
        PixelMatrixBUFF[i]=new Pixel[in.nHpix];
        for(int j=0;j<in.nHpix;j++)
        {
            PixelMatrixBUFF[i][j]=in(i,j);
        }
    }

    for(unsigned r = 0; r< REPS; ++r)
    {
#pragma omp parallel for schedule(dynamic)
        for(unsigned row=0; row<in.nVpix; ++row)
        {
            for(unsigned col=0; col<in.nHpix; ++col)
            {
                // transpose image coordinates to Cartesian coordinates
                unsigned h=in.nHpix/2;
                unsigned v=in.nVpix/2;	// integer div
                double X=(double)col-(double)h;
                double Y=(double)v-(double)row;

                // image rotation matrix
                double newX=cos(rotAngle)*X+sin(rotAngle)*Y;
                double newY=-sin(rotAngle)*X+cos(rotAngle)*Y;

                // Scale to fit everything in the image box
                double dH=(double)in.nHpix;
                double dV=(double)in.nVpix;
                double diagonal=sqrt(dH*dH+dV*dV);
                double ScaleFactor=(in.nHpix>in.nVpix) ? diagonal/dV : diagonal/dH;
                newX=newX*ScaleFactor;
                newY=newY*ScaleFactor;

                // convert back from Cartesian to image coordinates
                int rltCol0=(int) (newX+h);
                int rltRow0=(int)(v-newY);
                if((rltCol0>=0) && (rltRow0>=0) && (rltCol0<in.nHpix) && (rltRow0<in.nVpix)){
                    int rltCol1=rltCol0+1;
                    int rltRow1=rltRow0+1;
                    double tmpx=newX+h-rltCol0;
                    double tmpy=v-newY-rltRow0;
                    out(row, col).R=round((1-tmpx)*(1-tmpy)*PixelMatrixBUFF[rltRow0][rltCol0].R+(tmpx)*(1-tmpy)*PixelMatrixBUFF[rltRow0][rltCol1].R+(1-tmpx)*tmpy*PixelMatrixBUFF[rltRow1][rltCol0].R+(tmpx)*(tmpy)*PixelMatrixBUFF[rltRow1][rltCol1].R);
                    out(row, col).G=round((1-tmpx)*(1-tmpy)*PixelMatrixBUFF[rltRow0][rltCol0].G+(tmpx)*(1-tmpy)*PixelMatrixBUFF[rltRow0][rltCol1].G+(1-tmpx)*tmpy*PixelMatrixBUFF[rltRow1][rltCol0].G+(tmpx)*(tmpy)*PixelMatrixBUFF[rltRow1][rltCol1].G);
                    out(row, col).B=round((1-tmpx)*(1-tmpy)*PixelMatrixBUFF[rltRow0][rltCol0].B+(tmpx)*(1-tmpy)*PixelMatrixBUFF[rltRow0][rltCol1].B+(1-tmpx)*tmpy*PixelMatrixBUFF[rltRow1][rltCol0].B+(tmpx)*(tmpy)*PixelMatrixBUFF[rltRow1][rltCol1].B);
                }
            }
        }
    }
    return ; 
}

vector<string> arguments(int argc, char* argv[])
{
    vector<string> res;
    for (int i = 0; i!=argc; ++i)
        res.push_back(argv[i]);
    return res;
}

