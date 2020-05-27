/*
 * \file imageBMPcuda.cc
 */
#include "imageBMPcuda.hpp"

int CImageBMP::ReadBMP(string const &filename){
	fstream inFile(filename, ios::in | ios::binary);
	if(!inFile.is_open()){
		cout <<"\n\nFILE NOT FOUND: "<<filename <<"\n\n";
		return 1;
	}
    //cout <<sizeof(CPixel)<<" sizeof(CPixel) \n";
	// extract image height and width from header
    inFile.read(reinterpret_cast<char*> (&(this->HeaderInfo[0])), sizeof(unsigned char)*54);
    this->nHpix =  *(int*)&(this->HeaderInfo[18]);
	this->nVpix = *(int*)&(this->HeaderInfo[22]);
	cout <<"\nInput  BMP File name: "<<filename<<" ("<<this->nHpix<<","<<this->nVpix<<")\n";

    this->pixMat.resize(this->nVpix, this->nHpix); //this->nHpix * this->nVpix);
    unsigned long l = this->nVpix * this->nHpix;// *sizeof(CPixel);
	inFile.read(reinterpret_cast<char*>(this->pixMat.data()), l*sizeof(CPixel));
	inFile.close();
    return 0;
}

int CImageBMP::WriteBMP(string const &filename){
	fstream outFile(filename, ios::out | ios::binary);
	if(!outFile.is_open()){
		cout <<"\n\nFILE CREATION ERROR: "<<filename <<"\n\n";
		return 1;
	}
	//write header
    outFile.write(reinterpret_cast<char*>(this->HeaderInfo), sizeof(unsigned char)*54);
    unsigned long l = this->nVpix * this->nHpix;
	//write image data
    outFile.write(reinterpret_cast<char*>(this->pixMat.data()), l*sizeof(CPixel));
    outFile.close();
	cout <<"Output BMP File name: "<<filename <<" ("<<this->nHpix <<","<<nVpix<<")\n";  
    return 0;
}
