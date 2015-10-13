#ifndef IMAGEANALYZER_H
#define IMAGEANALYZER_H

#include "Headers.h"
#include "FacialFeatures.h"
class FacialFeatures;

class ImageAnalyzer
{

public:
    ImageAnalyzer();
    ~ImageAnalyzer();
    static void setFF(FacialFeatures* ff);
    static void findBestContour(Mat& src, vector<Point>& contour, Point offset, ROIType roi = NONE_ROI);
    static void findBestObject(Mat& src, Rect& dstROI, string dataPath);

    static void findEyePoints(Mat& src, ROIType roi);
    static void findEyebrowPoints(Mat& src, ROIType roi);
    static void findMouthPoints(Mat& src);

private:
    static FacialFeatures* ff;

};

#endif // IMAGEANALYZER_H
