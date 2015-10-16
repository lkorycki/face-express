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
    static void negateMat(const Mat& src, Mat& dst);
    static void clearBinBorder(const Mat& src, Mat& dst); // white to black
    static void clearGrayBorderV(const Mat &src, Mat &dst, int lw, float lmax, int rw = 0, float rmax = 0);
    static void clearGrayBorderH(const Mat &src, Mat &dst, int tw, float tmax, int bw = 0, float bmax = 0);

    // Specialized for each ROI
    static void createEyeMap(const Mat& src, Mat& dst);
    static void binarizeEye(const Mat& src, Mat& dst);
    static void binarizeEyebrow(const Mat& src, Mat& dst, float p, int ys);
    static void createMouthMap(const Mat& src, Mat& dst);
    static void binarizeMouth(const Mat& src, Mat& dst, float p);
    static void createMouthCornerMap(const Mat& src, Mat& dst, float p);
    static void binarizeTeeth(const Mat& src, Mat& dst, int t);

    // ROI processing pipelines
    static void preprocessEyeROI(const Mat& src, Mat& dst);
    static void preprocessEyebrowROI(const Mat& src, Mat& dst, ROIType roi);
    static void preprocessMouthROI(const Mat& src, Mat& dst);

private:
    static FacialFeatures* ff;
};

#endif // IMAGEPROCESSOR_H
