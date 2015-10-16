#ifndef IMAGEPROCESSOR_H
#define IMAGEPROCESSOR_H

#include "Headers.h"
#include "MathCore.h"
#include "FacialFeatures.h"
class FacialFeatures;

class ImageProcessor
{

public:
    ImageProcessor();
    ~ImageProcessor();
    static void setFF(FacialFeatures* ff); // for facial features arrays

    // General methods
    static void negateMat(Mat& src, Mat& dst);
    static void clearBinBorder(Mat& src, Mat& dst); // white to black
    static void clearGrayBorderV(Mat &src, Mat &dst, int lw, float lmax, int rw = 0, float rmax = 0);
    static void clearGrayBorderH(Mat &src, Mat &dst, int tw, float tmax, int bw = 0, float bmax = 0);

    // Specialized for each ROI
    static void createEyeMap(Mat& src, Mat& dst);
    static void binarizeEye(Mat& src, Mat& dst);
    static void binarizeEyebrow(Mat& src, Mat& dst, float p, int ys);
    static void createMouthMap(Mat& src, Mat& dst);
    static void createMouthCornerMap(Mat& src, Mat& dst, float p);
    static void binarizeMouth(Mat& src, Mat& dst, float p);
    static void binarizeTeeth(Mat& src, Mat& dst, int t);

    // ROI processing pipelines
    static void preprocessEyeROI(Mat& src, Mat& dst);
    static void preprocessEyebrowROI(Mat& src, Mat& dst, ROIType roi);
    static void preprocessMouthROI(Mat& src, Mat& dst);

private:
    static FacialFeatures* ff;
};

#endif // IMAGEPROCESSOR_H
