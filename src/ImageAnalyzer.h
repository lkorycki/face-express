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
    static void findBestContour(const Mat& src, vector<Point>& contour, Point offset, ROIType roi = NONE_ROI);
    static void findBestObject(const Mat& src, Rect& dstROI, string dataPath);
    static bool assertROI(const Mat& src, const Rect& roi);

    // Specialized
    static void findEyePoints(const Mat& src, ROIType roi);
    static void findEyebrowPoints(const Mat& src, ROIType roi);
    static void findMouthPoints(const Mat& src);

private:
    static FacialFeatures* ff;

};

#endif // IMAGEANALYZER_H
