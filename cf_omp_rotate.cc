/*
 * \file cf_omp_rotate.cc
 */
#include "cf_omp_rotate.hpp"

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

    // Initial a pixel for buff use
    //CPixel PixelBUFF;
    //PixelBUFF.R=0;
    //PixelBUFF.G=0;
    //PixelBUFF.B=0;

    // Initial a matrix for pixel save
    PixelDouble **PixelMatrixBUFF;
    PixelMatrixBUFF=new PixelDouble*[in.nVpix];
    for(int i=0;i<in.nVpix;i++)
    {
        PixelMatrixBUFF[i]=new PixelDouble[in.nHpix];
    }



    for(unsigned r = 0; r< REPS; ++r)
    {
        for(int i=0;i<in.nVpix;i++)
        {
            for(int j=0;j<in.nHpix;j++)
            {
                PixelMatrixBUFF[i][j].R=0;
                PixelMatrixBUFF[i][j].G=0;
                PixelMatrixBUFF[i][j].B=0;
                PixelMatrixBUFF[i][j].Weight=0;
            }
        }
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
                int rltCol0=(int) (newX+h);
                int rltRow0=(int)(v-newY);
                int rltCol1=rltCol0+1;
                int rltRow1=rltRow0+1;
                double tmpx=newX+h-rltCol0;
                double tmpy=v-newY-rltRow0;
                if((rltCol0>=0) && (rltRow0>=0) && (rltCol0<in.nHpix) && (rltRow0<in.nVpix)){
                    PixelMatrixBUFF[rltRow0][rltCol0].R +=(1-tmpx)*(1-tmpy)*in(row, col).R ;
                    PixelMatrixBUFF[rltRow0][rltCol0].G +=(1-tmpx)*(1-tmpy)*in(row, col).G ;
                    PixelMatrixBUFF[rltRow0][rltCol0].B +=(1-tmpx)*(1-tmpy)*in(row, col).B ;
                    PixelMatrixBUFF[rltRow0][rltCol1].R +=(tmpx)*(1-tmpy)*in(row, col).R ;
                    PixelMatrixBUFF[rltRow0][rltCol1].G +=(tmpx)*(1-tmpy)*in(row, col).G ;
                    PixelMatrixBUFF[rltRow0][rltCol1].B +=(tmpx)*(1-tmpy)*in(row, col).B ;
                    PixelMatrixBUFF[rltRow1][rltCol0].R +=(1-tmpx)*(tmpy)*in(row, col).R ;
                    PixelMatrixBUFF[rltRow1][rltCol0].G +=(1-tmpx)*(tmpy)*in(row, col).G ;
                    PixelMatrixBUFF[rltRow1][rltCol0].B +=(1-tmpx)*(tmpy)*in(row, col).B ;
                    PixelMatrixBUFF[rltRow1][rltCol1].R +=(tmpx)*(tmpy)*in(row, col).R ;
                    PixelMatrixBUFF[rltRow1][rltCol1].G +=(tmpx)*(tmpy)*in(row, col).G ;
                    PixelMatrixBUFF[rltRow1][rltCol1].B +=(tmpx)*(tmpy)*in(row, col).B ;
                    PixelMatrixBUFF[rltRow0][rltCol0].Weight +=(1-tmpx)*(1-tmpy);
                    PixelMatrixBUFF[rltRow0][rltCol1].Weight +=(tmpx)*(1-tmpy);
                    PixelMatrixBUFF[rltRow1][rltCol0].Weight +=(1-tmpx)*(tmpy);
                    PixelMatrixBUFF[rltRow1][rltCol1].Weight +=(tmpx)*(tmpy);
                }
            }
        }
    }
#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<in.nVpix;i++)
    {
        for(int j=0;j<in.nHpix;j++)
        {
            PixelMatrixBUFF[i][j].R=PixelMatrixBUFF[i][j].R/PixelMatrixBUFF[i][j].Weight;
            PixelMatrixBUFF[i][j].G=PixelMatrixBUFF[i][j].G/PixelMatrixBUFF[i][j].Weight;
            PixelMatrixBUFF[i][j].B=PixelMatrixBUFF[i][j].B/PixelMatrixBUFF[i][j].Weight;
            out(i,j).R=(char)PixelMatrixBUFF[i][j].R;
            out(i,j).G=(char)PixelMatrixBUFF[i][j].G;
            out(i,j).B=(char)PixelMatrixBUFF[i][j].B;
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

