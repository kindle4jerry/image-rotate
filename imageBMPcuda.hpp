/*
 * \file imageBMPcuda.hpp
 */
#ifndef _IMAGEBMPCUDA_H_
#define _IMAGEBMPCUDA_H_
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cassert>
#include <Eigen/Eigen>
unsigned int const REPS = 121;

using namespace std;
typedef struct Pixel{
        unsigned char R;
        unsigned char G;
        unsigned char B;

} CPixel;

using MatrixXpix= Eigen::Matrix<CPixel, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
class CImageBMP {
    public:
        unsigned int nVpix;       // row, height of the image
        unsigned int nHpix;       // column, widith of the image
        unsigned char HeaderInfo[54];
        MatrixXpix pixMat;
        int ReadBMP(string const &filename);
        int WriteBMP(string const &filename);
};
#endif
