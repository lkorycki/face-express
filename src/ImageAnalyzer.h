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
    static void setFF(FacialFeatures* ff); // for facial features arrays

    // General methods
    static void findCorners(vector<Point> contour, Point& p1, Point& p2, bool horizontal);
    static void findBestContour(Mat& src, vector<Point>& contour, Point offset, ROIType roi = NONE_ROI);
    static void findBestObject(Mat& src, Rect& dstROI, string dataPath);

    // Specialized
    static void findEyePoints(Mat& src, ROIType roi);
    static void findEyebrowPoints(Mat& src, ROIType roi);
    static void findMouthPoints(Mat& src);

private:
    static FacialFeatures* ff;

};

#endif // IMAGEANALYZER_H
