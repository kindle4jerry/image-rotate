/*
 * \file gb_omp_rotate.cu
 */
#include "gb_omp_rotate.hpp"
#include <cuda_runtime.h>
#include "timer.hpp"
#define CHECK(res) if(res!=cudaSuccess){exit(-1);}
cudaError_t res;

__global__
void GetImage(int *PixelINR, int *PixelING, int *PixelINB, int *PixelOUTR, int *PixelOUTG, int *PixelOUTB, int inH, int inV, int REPS_dev, double SRAS_dev, double CRAS_dev, unsigned h_dev, unsigned v_dev){
    //for(unsigned r = 0; r< REPS_dev; ++r)
    {
        unsigned row=blockIdx.x;
        unsigned col=threadIdx.x;

        //for(unsigned row=0; row<inV; ++row)
        {
            double Y=(double)v_dev-(double)row;
            double SRAYS_dev=SRAS_dev*Y; 
            double CRAYS_dev=CRAS_dev*Y;
            //for(unsigned col=0; col<inH; ++col)
            {
                // transpose image coordinates to Cartesian coordinates
                double X=(double)col-(double)h_dev;

                double newX=CRAS_dev*X+SRAYS_dev;
                double newY=-SRAS_dev*X+CRAYS_dev;

                // convert back from Cartesian to image coordinates
                int rltCol0=(int) (newX+h_dev);
                int rltRow0=(int)(v_dev-newY);
                if((rltCol0>=0) && (rltRow0>=0) && (rltCol0<inH) && (rltRow0<inV)){
                    int rltCol1=rltCol0+1;
                    int rltRow1=rltRow0+1;
                    double tmpx=newX+h_dev-rltCol0;
                    double tmpy=v_dev-newY-rltRow0;
                    PixelOUTR[row*inH+col]=round((1-tmpx)*(1-tmpy)*PixelINR[rltRow0*inH+rltCol0]+(tmpx)*(1-tmpy)*PixelINR[rltRow0*inH+rltCol1]+(1-tmpx)*tmpy*PixelINR[rltRow1*inH+rltCol0]+(tmpx)*(tmpy)*PixelINR[rltRow1*inH+rltCol1]);
                    PixelOUTG[row*inH+col]=round((1-tmpx)*(1-tmpy)*PixelING[rltRow0*inH+rltCol0]+(tmpx)*(1-tmpy)*PixelING[rltRow0*inH+rltCol1]+(1-tmpx)*tmpy*PixelING[rltRow1*inH+rltCol0]+(tmpx)*(tmpy)*PixelING[rltRow1*inH+rltCol1]);
                    PixelOUTB[row*inH+col]=round((1-tmpx)*(1-tmpy)*PixelINB[rltRow0*inH+rltCol0]+(tmpx)*(1-tmpy)*PixelINB[rltRow0*inH+rltCol1]+(1-tmpx)*tmpy*PixelINB[rltRow1*inH+rltCol0]+(tmpx)*(tmpy)*PixelINB[rltRow1*inH+rltCol1]);
                    printf("%d %d %d %d %d\n",row,col,PixelOUTR[row*inH+col],PixelOUTG[row*inH+col],PixelOUTB[row*inH+col]);
                }
            }
        }
    }
    return;
}
void c_omp_rotate(CImageBMP &in, double const &rotAngle, CImageBMP &out)
{
    Timer t;
    double dH=(double)in.nHpix;
    double dV=(double)in.nVpix;
    double diagonal=sqrt(dH*dH+dV*dV);
    double ScaleFactor=(in.nHpix>in.nVpix) ? diagonal/dV : diagonal/dH;
    unsigned h=in.nHpix/2;
    unsigned v=in.nVpix/2;	// integer division
    unsigned innHpix=(unsigned)in.nHpix;
    unsigned innVpix=(unsigned)in.nVpix;

    double CRAS =cos(rotAngle)*ScaleFactor;	
    double SRAS =sin(rotAngle)*ScaleFactor;	
    t.printDiff("A1 time: ");
    int *PixelR=(int*)malloc(sizeof(int)*innHpix*innVpix);
    int *PixelG=(int*)malloc(sizeof(int)*innHpix*innVpix);
    int *PixelB=(int*)malloc(sizeof(int)*innHpix*innVpix);
    t.printDiff("A2 time: ");

    Pixel **PixelMatrixBUFF;
    PixelMatrixBUFF=new Pixel*[innVpix];
#pragma omp parallel for schedule(dynamic)
    for(int i=0;i<innVpix;i++)
    {
        PixelMatrixBUFF[i]=new Pixel[innHpix];
        for(int j=0;j<innHpix;j++)
        {
            PixelMatrixBUFF[i][j]=in(i,j);
            PixelR[i*innHpix+j]=PixelMatrixBUFF[i][j].R;
            PixelG[i*innHpix+j]=PixelMatrixBUFF[i][j].G;
            PixelB[i*innHpix+j]=PixelMatrixBUFF[i][j].B;
        }
    }

    t.printDiff("A3 time: ");
    int NumGPUs = 0;
    cudaGetDeviceCount(&NumGPUs);
    if (NumGPUs == 0){
		cout <<"\nNo CUDA Device is available\n\n";
		return; 
    }
    cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n\n";
		return;
	}
	
    cudaDeviceProp GPUprop;
    cudaGetDeviceProperties(&GPUprop, 0);
    

    size_t const IMAGEPIX = (innHpix*innVpix);
    size_t const IMAGESIZE = 3*IMAGEPIX*sizeof(int);
    size_t GPUtotalBufferSize =2*IMAGESIZE;
    t.printDiff("A3.3 time: ");
    
    void *ptrGPU;			// Pointer to the bulk-allocated GPU memory
    cudaStatus=cudaMalloc((void**)&ptrGPU, GPUtotalBufferSize);
    if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaMallocfailed\n\n";
		return;
	}
    t.printDiff("A4 time: ");
    int *PixelTransR=(int *)ptrGPU;
    int *PixelTransG=PixelTransR+IMAGEPIX;
    int *PixelTransB=PixelTransG+IMAGEPIX;
    int *PixelTrans2R=PixelTransB+IMAGEPIX;
    int *PixelTrans2G=PixelTrans2R+IMAGEPIX;
    int *PixelTrans2B=PixelTrans2G+IMAGEPIX;
    t.printDiff("A5 time: ");
    cudaMemcpy((void*)(PixelTransR),(void*)(PixelR),innHpix*innVpix*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy((void*)(PixelTransG),(void*)(PixelG),innHpix*innVpix*sizeof(int),cudaMemcpyHostToDevice);
    cudaStatus=cudaMemcpy((void*)(PixelTransB),(void*)(PixelB),innHpix*innVpix*sizeof(int),cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaMallocfailed\n\n";
		return;
	}
    t.printDiff("A6 time: ");

    dim3 grid(innVpix,1,1);
    dim3 thread(innHpix,1,1);
    t.printDiff("A7 time: ");
    
    
    GetImage<<<grid,thread>>>(PixelTransR,PixelTransG,PixelTransB,PixelTrans2R,PixelTrans2G,PixelTrans2B,innHpix,innVpix,REPS,SRAS,CRAS,h, v);
    t.printDiff("GPU kernel time: ");

    cudaMemcpy((void*)(PixelR), (void*)(PixelTrans2R),innHpix*innVpix*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)(PixelG), (void*)(PixelTrans2G),innHpix*innVpix*sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)(PixelB), (void*)(PixelTrans2B),innHpix*innVpix*sizeof(int), cudaMemcpyDeviceToHost);
    t.printDiff("A8 time: ");

#pragma omp parallel for schedule(dynamic)
    for(unsigned i=0;i<innVpix;i++)
    {
        for(unsigned j=0;j<innHpix;j++)
        {
            PixelMatrixBUFF[i][j].R=(char)PixelR[i*innHpix+j];
            PixelMatrixBUFF[i][j].G=(char)PixelG[i*innHpix+j];
            PixelMatrixBUFF[i][j].B=(char)PixelB[i*innHpix+j];
            printf("%d %d %d %d %d\n",i,j,PixelMatrixBUFF[i][j].R,PixelMatrixBUFF[i][j].G,PixelMatrixBUFF[i][j].B);
            out(i,j)=PixelMatrixBUFF[i][j];
        }
    }
    t.printDiff("A9 time: ");
    return ; 
}

vector<string> arguments(int argc, char* argv[])
{
    vector<string> res;
    for (int i = 0; i!=argc; ++i)
        res.push_back(argv[i]);
    return res;
}

