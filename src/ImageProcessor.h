#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include "Headers.h"
#include "MathCore.h"

class ImageProcessor
{

public:
    ImageProcessor();
    static void negateMat(Mat& src, Mat& dst);
    static void clearBinBorder(Mat& src, Mat& dst); // default: white to black
    static void clearGrayBorderV(Mat &src, Mat &dst, int lw, float lmax, int rw = 0, float rmax = 0);
    static void clearGrayBorderH(Mat &src, Mat &dst, int tw, float tmax, int bw = 0, float bmax = 0);
    static void findCorners(vector<Point> contour, Point& p1, Point& p2, bool horizontal = true);
    static void createMouthCornerMap(Mat& src, Mat& dst, float p);

    static void createEyeMap(Mat& src, Mat& dst);
    static void binarizeEye(Mat& src, Mat& dst);
    static void binarizeEyebrow(Mat& src, Mat& dst, float p, int ys);
    static void createMouthMap(Mat& src, Mat& dst);
    static void binarizeMouth(Mat& src, Mat& dst, float p);
    static void binarizeTeeth(Mat& src, Mat& dst, int t);
};

#endif // IMAGEPROCESSOR_H
