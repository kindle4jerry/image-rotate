/*
 * \file omp_rotate.cc
 */
#include "omp_rotate.hpp"

void omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out)
{
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
                double newX=cos(rotAngle)*X-sin(rotAngle)*Y;
                double newY=sin(rotAngle)*X+cos(rotAngle)*Y;

                // Scale to fit everything in the image box
                double dH=(double)in.nHpix;
                double dV=(double)in.nVpix;
                double diagonal=sqrt(dH*dH+dV*dV);
                double ScaleFactor=(in.nHpix>in.nVpix) ? dV/diagonal : dH/diagonal;
                newX=newX*ScaleFactor;
                newY=newY*ScaleFactor;
                
                // convert back from Cartesian to image coordinates
                int rltCol=((int) newX+h);
                int rltRow=v-(int)newY;
                if((rltCol>=0) && (rltRow>=0) && (rltCol<in.nHpix) && (rltRow<in.nVpix)){
                    out(rltRow, rltCol) = in(row, col);
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

