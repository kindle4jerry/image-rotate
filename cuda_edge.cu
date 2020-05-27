/*
 * \file cuda_edge.cu
 */
#include "cuda_edge.hpp"
#include "timer.hpp"
__device__
inline unsigned un_ceil(unsigned const a, unsigned const b) {return (a+b-1)/b; }

// Kernel that calcsize_tates a B&W image from an RGB image
// ressize_tting image has a double type for each pixel position
__global__
void bw_kernel(double *bwMat, unsigned char *gpuMat, unsigned nHpix)
{
	unsigned nThdsPerBlk = blockDim.x;
	unsigned bIdx = blockIdx.x;
	unsigned tIdx = threadIdx.x;
	unsigned gtIdx = nThdsPerBlk * bIdx + tIdx;
	double R, G, B;

	unsigned BlkPerRow = un_ceil(nHpix, nThdsPerBlk);
	unsigned RowBytes = (nHpix * 3 + 3) & (~3);
	unsigned rowBgn = bIdx / BlkPerRow;
	unsigned colBgn = gtIdx - rowBgn*BlkPerRow*nThdsPerBlk;
	if (colBgn >= nHpix) return;			// col out of range

	unsigned srcIdx = rowBgn * RowBytes + 3 * colBgn;
	unsigned pixIdx = rowBgn * nHpix + colBgn;

	B = (double)gpuMat[srcIdx];
	G = (double)gpuMat[srcIdx + 1];
	R = (double)gpuMat[srcIdx + 2];
	bwMat[pixIdx] = (R+G+B)/3.0;
}


int launch_edge_kernel(CImageBMP &image, unsigned const ThreshLo, unsigned const ThreshHi, unsigned const nThdsPerBlk) 
{
	Timer t;
	// Choose which GPU to run on, change this on a msize_tti-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		cout <<"\nNo CUDA Device is available\n\n";
		return 1; 
	}
	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaSetDevice failed!  Do you have a CUDA-capable GPU installed?\n\n";
		return 1;
	}
	
    cudaDeviceProp GPUprop;
	cudaGetDeviceProperties(&GPUprop, 0);
	size_t SupportedKBlocks = (unsigned) GPUprop.maxGridSize[0] * (unsigned) GPUprop.maxGridSize[1] * (unsigned )GPUprop.maxGridSize[2]/1024;
	size_t SupportedMBlocks = SupportedKBlocks / 1024;
    size_t outB =  (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks;
    char outC = (SupportedMBlocks >= 5) ? 'M' : 'K';
    string SupportedBlocks(to_string(outB)); 
    SupportedBlocks = SupportedBlocks +" " + outC; 
	size_t maxThdsPerBlk = (unsigned)GPUprop.maxThreadsPerBlock;

	cudaEvent_t time1, time2, time2BW, time2Gauss, time2Sobel, time3, time4;
	cudaEventCreate(&time1);		cudaEventCreate(&time2);	
	cudaEventCreate(&time2BW);		cudaEventCreate(&time2Gauss);	cudaEventCreate(&time2Sobel);	
	cudaEventCreate(&time3);		cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer
	
	
	// Allocate GPU buffer for the input and output images and the intermediate result
    size_t const IMAGEPIX  = (image.nHpix*image.nVpix);
    size_t const IMAGESIZE = IMAGEPIX* sizeof(CPixel); //image.nHpix * image.nVpix * sizeof(CPixel); 
    size_t GPUtotalBufferSize = 2 * sizeof(unsigned char)*IMAGESIZE;
	t.printDiff("Init time: ");
	
	void *ptrGPU;			// Pointer to the bulk-allocated GPU memory
    cudaStatus = cudaMalloc((void**)&ptrGPU, GPUtotalBufferSize);
    if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaMalloc failed! Can't allocate GPU memory\n\n";
		return 1; 
	}
	t.printDiff("Malloc time: ");
	unsigned char *imgGPU, *ptrImgGPU;	// Where images are stored in GPU
	imgGPU			= (unsigned char *)ptrGPU;
	ptrImgGPU	= imgGPU + IMAGESIZE;

    double  *GPUBWImg, *GPUGaussImg, *gradGPU, *thetaGPU;
	GPUBWImg	= (double *)(ptrImgGPU + IMAGESIZE);
	GPUGaussImg	= GPUBWImg + IMAGEPIX;
	gradGPU		= GPUGaussImg + IMAGEPIX;
	thetaGPU	= gradGPU + IMAGEPIX;

	t.printDiff("Memcpystart time: ");
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(imgGPU, image.pixMat.data(), IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaMemcpy  CPU to GPU  failed!\n\n";
		return 1; 
	}
	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	t.printDiff("Memcpy1 time: ");
	
	unsigned BlkPerRow = host_ceil(image.nHpix, nThdsPerBlk);
	unsigned nBlks = image.nVpix*BlkPerRow;
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	bw_kernel <<< nBlks, nThdsPerBlk >>> (GPUBWImg, imgGPU, image.nHpix);
	if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) return 1; //goto KERNELERROR;
	cudaEventRecord(time2BW, 0);		// Time stamp after BW image calcsize_tation
	unsigned GPUDataTfrBW, GPUDataTfrGauss, GPUDataTfrSobel, GPUDataTfrThresh,GPUDataTfrKernel, GPUDataTfrTotal;
	GPUDataTfrBW = sizeof(double)*IMAGEPIX + sizeof(unsigned char)*IMAGESIZE;

	//gauss_kernel <<< nBlks, nThdsPerBlk >>> (GPUGaussImg, GPUBWImg, image.nHpix, image.nVpix);
	//if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) return 1; //goto KERNELERROR; 
	cudaEventRecord(time2Gauss, 0);		// Time stamp after Gauss image calcsize_tation
	GPUDataTfrGauss = 2*sizeof(double)*IMAGEPIX;

	//sobel_kernel <<< nBlks, nThdsPerBlk >>> (gradGPU, thetaGPU, GPUGaussImg, image.nHpix, image.nVpix);
	//if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) return 1; //goto KERNELERROR; 
	cudaEventRecord(time2Sobel, 0);		// Time stamp after Gradient, Theta computation
	GPUDataTfrSobel = 3 * sizeof(double)*IMAGEPIX;

	//prune_kernel <<< nBlks, nThdsPerBlk >>> (ptrImgGPU, gradGPU, thetaGPU, image.nHpix, image.nVpix, ThreshLo, ThreshHi);
	//if ((cudaStatus = cudaDeviceSynchronize()) != cudaSuccess) return 1; //goto KERNELERROR;
	GPUDataTfrThresh = sizeof(double)*IMAGEPIX + sizeof(unsigned char)*IMAGESIZE;
	GPUDataTfrKernel = GPUDataTfrBW + GPUDataTfrGauss + GPUDataTfrSobel + GPUDataTfrThresh;
	GPUDataTfrTotal = GPUDataTfrKernel + 2 * IMAGESIZE;
	cudaEventRecord(time3, 0);

	t.printDiff("Memcpystart time: ");
	// Copy output (ressize_tts) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(image.pixMat.data(), ptrImgGPU, IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		cout <<"\ncudaMemcpy GPU to CPU  failed! \n\n";
		return 1; //goto EXITCUDAERROR;
	}
	cudaEventRecord(time4, 0);
	t.printDiff("Memcpy2 time: ");

	cudaEventSynchronize(time1);	cudaEventSynchronize(time2);
	cudaEventSynchronize(time2BW);	cudaEventSynchronize(time2Gauss);	cudaEventSynchronize(time2Sobel);
	cudaEventSynchronize(time3);	cudaEventSynchronize(time4);

    float totalTime, tfrCPUtoGPU, tfrGPUtoCPU;
	float kernelExecTimeBW, kernelExecTimeGauss, kernelExecTimeSobel, kernelExecTimeThreshold;
	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecTimeBW, time2, time2BW);
	cudaEventElapsedTime(&kernelExecTimeGauss, time2BW, time2Gauss);
	cudaEventElapsedTime(&kernelExecTimeSobel, time2Gauss, time2Sobel);
	cudaEventElapsedTime(&kernelExecTimeThreshold, time2Sobel, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);
	float totalKernelTime = kernelExecTimeBW + kernelExecTimeGauss + kernelExecTimeSobel + kernelExecTimeThreshold;

	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		cout <<"\n Program failed after cudaDeviceSynchronize()!\n\n";
		return 1; 
	}
	cout <<"Computation configuration: Threads per block="<<nThdsPerBlk<<", ["<<nBlks<<" BLOCKS, "<< BlkPerRow << " BLOCKS/ROW]\n";
	cout <<"\n--------------------------------------------------------------------------\n";
	cout <<GPUprop.name <<"    Compute Capability "<<GPUprop.major<<"."<<GPUprop.minor<<"  [max "<<SupportedBlocks<<" blocks; "<<maxThdsPerBlk<<" thr/blk] \n"; 
	cout <<"--------------------------------------------------------------------------\n"<<fixed<<setw(10)<<setprecision(2);
	cout <<"--------------------------------------------------------------------------\n";
	cout <<"            CPU->GPU Transfer   ="<<tfrCPUtoGPU<<" ms  ...  "<<set_size_MB(IMAGESIZE)<<" MB  ...  "<<setw(8)<<set_BW_GB(IMAGESIZE, tfrCPUtoGPU)<<" GB/s\n";
	cout <<"            GPU->CPU Transfer   ="<<tfrGPUtoCPU<<" ms  ...  "<<set_size_MB(IMAGESIZE)<<" MB  ...  "<<setw(8)<<set_BW_GB(IMAGESIZE, tfrGPUtoCPU)<<" GB/s\n";
	cout <<"----------------------------------------------------------------------------\n";
	cout <<"       BW Kernel Execution Time ="<<kernelExecTimeBW<<" ms  ...  "<<set_size_MB(GPUDataTfrBW)<<" MB  ...  "<< set_BW_GB(GPUDataTfrBW, kernelExecTimeBW)<<" GB/s\n";
	cout <<"    Gauss Kernel Execution Time ="<<kernelExecTimeGauss<<" ms  ...  "<<set_size_MB(GPUDataTfrGauss)<<" MB  ...  "<< set_BW_GB(GPUDataTfrGauss, kernelExecTimeGauss)<<" GB/s\n";
	cout <<"    Sobel Kernel Execution Time ="<<kernelExecTimeSobel<<" ms  ...  "<<set_size_MB(GPUDataTfrSobel)<<" MB  ...  "<< set_BW_GB(GPUDataTfrSobel, kernelExecTimeSobel)<<" GB/s\n";
	cout <<"Threshold Kernel Execution Time ="<<kernelExecTimeThreshold<<" ms  ...  "<<set_size_MB(GPUDataTfrThresh)<<" MB  ...  "<< set_BW_GB(GPUDataTfrThresh, kernelExecTimeThreshold)<<" GB/s\n";
	cout <<"----------------------------------------------------------------------------\n";
	cout <<"Total GPU kernel-only time      ="<<totalKernelTime<<" ms       "<< set_size_MB(GPUDataTfrKernel)<<" MB  ...  "<< setw(8)<<set_BW_GB(GPUDataTfrKernel, totalKernelTime)<<" GB/s\n";
	cout <<"Total time with I/O included    ="<<totalTime <<     " ms  ...  "<< set_size_MB(GPUDataTfrTotal) <<" MB  ...  "<<set_BW_GB(GPUDataTfrTotal, totalTime)<<" GB/s\n"; 
	cout <<"----------------------------------------------------------------------------\n";

	// Deallocate GPU memory and destroy events.
	cudaFree(ptrGPU);
	cudaEventDestroy(time1);	cudaEventDestroy(time2);
	cudaEventDestroy(time2BW);	cudaEventDestroy(time2Gauss);	cudaEventDestroy(time2Sobel);
	cudaEventDestroy(time3);	cudaEventDestroy(time4);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools sunsigned char as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
        cout <<"\ncudaDeviceReset failed!\n\n";
        return 1;
	}
    return 0;

}
