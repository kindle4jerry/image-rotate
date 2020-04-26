/*
 * \file drv_c_omp_rotate.cc
 */
#include "imageBMP.hpp"
#include "c_omp_rotate.hpp"
#include "timer.hpp"
unsigned int const MAXTHREADS = 72;

int main(int argc, char** argv) {
    unsigned nThreads = 1;
    double rotDegrees = 45;
    vector<string> strArgv = arguments(argc, argv);
    switch (argc){
        case 3 : 						break;
        case 4 : rotDegrees = stoi(strArgv[3]);		break;
        case 5 : nThreads=stoi(strArgv[4]);  rotDegrees = stoi(strArgv[3]);		break;
        default: cout <<"\n\nUsage: "<<strArgv[0]<<" input output [rotate_angle] [thread count]\n";
                 cout <<"\nExample: "<<strArgv[0]<<" infilename.bmp outname.bmp 45 8\n\n";
                 cout <<"\nExample: "<<strArgv[0]<<" infilename.bmp outname.bmp -65\n\n";
                 return 0;
    }
    if((rotDegrees < -360 ) && (rotDegrees > 360)) {
        cout <<"Rotation angle of "<<rotDegrees<<" is invalid. \n Please use an angle between -360 and 360 ...\n";
        exit(EXIT_FAILURE);
    }

    if((nThreads<1) || (nThreads>MAXTHREADS)){
        cout <<"\nNumber of threads must be between 1 and "<<MAXTHREADS<<" ... Exiting abruptly\n";
        exit(EXIT_FAILURE);
    }
    else{
        if(nThreads != 1){
            cout <<"\nExecuting the multi-threaded version with "<<nThreads<<" threads ...\n";
        }
        else{
            cout <<"\nExecuting the serial version ...\n";
        }
    }
	double rotAngle=2*3.141592/360.000*(double) rotDegrees;   // Convert the angle to radians
	cout <<"\nRotating image by "<< rotDegrees <<" degrees ("<<rotAngle<<" radians) ...\n"; 

    CImageBMP imag;
    imag.ReadBMP(strArgv[1]);
    CImageBMP rotated(imag.nVpix, imag.nHpix);
    rotated.set_header(imag.HeaderInfo);
    Timer t;
    omp_set_num_threads(nThreads);
    c_omp_rotate(imag, rotAngle, rotated); 
    rotated.WriteBMP(strArgv[2]);
    double duration = t.printDiff("Execution time: ");
    cout <<"("<<duration*1000000/(double)(imag.nHpix*imag.nVpix*REPS)<<" ns per pixel)\n\n";

    return 0;
}
