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

int main(int argc, char **argv)
{
    cudaSetDevice(0);
    cudaFree(0);
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
