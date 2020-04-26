/*
 * \file c_omp_rotate.cc
 */
#include "c_omp_rotate.hpp"

void c_omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out)
{
    double dH=(double)in.nHpix;
    double dV=(double)in.nVpix;
    double diagonal=sqrt(dH*dH+dV*dV);
    double ScaleFactor=(in.nHpix>in.nVpix) ? dV/diagonal : dH/diagonal;
    unsigned h=in.nHpix/2;
    unsigned v=in.nVpix/2;	// integer division

    double CRAS =cos(rotAngle)*ScaleFactor;	
    double SRAS =sin(rotAngle)*ScaleFactor;	
    for(unsigned r = 0; r< REPS; ++r)
    {
#pragma omp parallel for schedule(dynamic)
        for(unsigned row=0; row<in.nVpix; ++row)
        {
            double Y=(double)v-(double)row;
            double SRAYS=SRAS*Y;     
            double CRAYS=CRAS*Y;
            for(unsigned col=0; col<in.nHpix; ++col)
            {
                // transpose image coordinates to Cartesian coordinates
                double X=(double)col-(double)h;

                double newX=CRAS*X-SRAYS;
                double newY=SRAS*X+CRAYS;

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

