/*
 * \file drv_edgeG.cc
 */

#include "imageBMPcuda.hpp"
#include "timer.hpp"
#include "cuda_edge.hpp"

vector<string> arguments(int argc, char* argv[])
{
    vector<string> res;
    for (int i = 0; i!=argc; ++i)
        res.push_back(argv[i]);
    return res;
}

bool InitCUDA()  
{  
    int count;  
    cudaGetDeviceCount(&count);  
    if(count==0)  
    {  
        fprintf(stderr,"There is no device.\n");  
        return false;  
    }  
    int i;  
    for(i=0;i<count;i++)  
    {  
        cudaDeviceProp prop;  
        if(cudaGetDeviceProperties(&prop,i) == cudaSuccess)  
        {  
            if(prop.major>=1)  
            {  
                //枚举详细信息  
                printf("Identify: %s\n",prop.name);  
                printf("Host Memory: %d\n",prop.canMapHostMemory);                  
                printf("Clock Rate: %d khz\n",prop.clockRate);                  
                printf("Compute Mode: %d\n",prop.computeMode);                  
                printf("Device Overlap: %d\n",prop.deviceOverlap);                  
                printf("Integrated: %d\n",prop.integrated);                  
                printf("Kernel Exec Timeout Enabled: %d\n",prop.kernelExecTimeoutEnabled);                  
                printf("Max Grid Size: %d * %d * %d\n",prop.maxGridSize[0],prop.maxGridSize[1],prop.maxGridSize[2]);  
                printf("Max Threads Dim: %d * %d * %d\n",prop.maxThreadsDim[0],prop.maxThreadsDim[1],prop.maxThreadsDim[2]);  
                printf("Max Threads per Block: %d\n",prop.maxThreadsPerBlock);  
                printf("Maximum Pitch: %d bytes\n",prop.memPitch);  
                printf("Minor Compute Capability: %d\n",prop.minor);  
                printf("Number of Multiprocessors: %d\n",prop.multiProcessorCount);                  
                printf("32bit Registers Availble per Block: %d\n",prop.regsPerBlock);  
                printf("Shared Memory Available per Block: %d bytes\n",prop.sharedMemPerBlock);  
                printf("Alignment Requirement for Textures: %d\n",prop.textureAlignment);  
                printf("Constant Memory Available: %d bytes\n",prop.totalConstMem);  
                printf("Global Memory Available: %d bytes\n",prop.totalGlobalMem);  
                printf("Warp Size: %d threads\n",prop.warpSize);  
                break;  
            }  
        }  
    }  
    if(i==count)  
    {  
        fprintf(stderr,"There is no device supporting CUDA.\n");  
        return false;  
    }  
    cudaSetDevice(i);  
    return true;  
}  

int main(int argc, char **argv)
{
    InitCUDA();

    unsigned 	ThreshLo=50, ThreshHi=100;		// "Edge" vs. "No Edge" thresholds
    vector<string> strArgv = arguments(argc, argv);
    unsigned nThdsPerBlk=256;
	switch (argc){
		case 6:  ThreshHi  = stoi(strArgv[5]);
		case 5:  ThreshLo  = stoi(strArgv[4]);
		case 4:  nThdsPerBlk = stoi(strArgv[3]);
		case 3:  break;
		default: cout <<"\n\nUsage:   "<<strArgv[0]<<"InputFile OutputFile [nThdsPerBlk] [ThreshLo] [ThreshHi]" ;
				 cout <<"\n\nExample: "<<strArgv[0]<<" car_racing.bmp Output.bmp"; 
				 cout <<"\n\nExample: "<<strArgv[0]<<" car_racing.bmp Output.bmp 256";
				 cout <<"\n\nExample: "<<strArgv[0]<<" car_racing.bmp Output.bmp 256 50 100\n\n";
				 return 0 ;
	}
	if ((nThdsPerBlk < 32) || (nThdsPerBlk > 1024)) {
		cout <<"Invalid nThdsPerBlk option "<< nThdsPerBlk<<". Must be between 32 and 1024. \n";
				 return 0 ;
	}
	if ((ThreshLo<0) || (ThreshHi>255) || (ThreshLo>ThreshHi)){
		cout <<"\nInvalid Thresholds: Threshold must be between [0...255] ...\n";
		cout <<"\n\nNothing executed ... Exiting ...\n\n";
				 return 0 ;
	}

    // Create CPU memory to store the input and output images
    Timer t;
    CImageBMP imag;
    int err = imag.ReadBMP(strArgv[1]);
    if (err != 0){
        cout <<"Cannot allocate memory for the input image...\n";
        return 0;
    }
    t.printDiff("Reading BMP file time: ");

    cout <<strArgv[0]<<" "<<strArgv[1]<<" "<<strArgv[2]<<", ThreshLo= "<<ThreshLo<<", ThreshHi="<<ThreshHi <<"\n";
    err = launch_edge_kernel(imag, ThreshLo, ThreshHi, nThdsPerBlk);

    if(err ==0){
        imag.WriteBMP(strArgv[2]);
        t.printDiff("Total Execution time: ");
        return 0;
    }
    return 0;
}
