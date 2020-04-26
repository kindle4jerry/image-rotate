/*
 * \file imageBMP.hpp
 */
#ifndef _IMAGEBMP_H_
#define _IMAGEBMP_H_
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
unsigned int const REPS = 11;

using namespace std;
typedef struct Pixel{
        unsigned char R;
        unsigned char G;
        unsigned char B;

} CPixel;

class CImageBMP {
    public:
        unsigned int nVpix;       // row, height of the image
        unsigned int nHpix;       // column, widith of the image
        unsigned char HeaderInfo[54];
        std::vector<CPixel> pixMat;

    public:
        CImageBMP(const unsigned v=0, const unsigned h=0):nVpix(v), nHpix(h){
            if(v*h !=0) pixMat.resize(v*h);
        }
        void set_header(unsigned char in[54]){std::copy(in, in+54, HeaderInfo);}
        CPixel operator() (unsigned int const row, unsigned int const col) const {
            assert(row < this->nVpix && col < this->nHpix);
            return pixMat[row*this->nHpix + col]; 
        } 
        CPixel& operator()(unsigned int const row, unsigned int const col){
            assert(row < this->nVpix && col < this->nHpix);
            return pixMat[row*this->nHpix + col]; 
        }
        void ReadBMP(string const &filename);
        void WriteBMP(string const &filename);
};
#endif
