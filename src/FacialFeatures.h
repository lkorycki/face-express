#ifndef FacialFeatures_H
#define FacialFeatures_H

#define ROI_NUM 8
#define OFF_NUM 6
#define FEAT_POINTS 20
#define FEAT_NUM 20


#include "Headers.h"

enum ROItype { L_EYE, R_EYE, L_EB, R_EB, MOUTH, NOSE, TEETH, FACE, NONE };

class FacialFeatures
{

public:
    FacialFeatures();
    ~FacialFeatures();
    void detectFace(Mat& src, Mat& dst);
    double* extractFacialFeatures(Mat& src);
    void setROI(Mat& src);

private:
    Mat ROI[ROI_NUM];
    Point* featurePoints;
    double* featureVector;
    int featPointOffsets[OFF_NUM];
    int featVecOffsets[OFF_NUM];

    void findBestContour(Mat& src, vector<Point>& contour, Point offset, ROItype roi = NONE);
    void findBestObject(Mat& src, Rect& dstROI, string dataPath);

    void extractEyesPoints();
    void preprocessEyeROI(Mat& src, Mat& dst);
    void findEyePoints(Mat& src, ROItype roi);

    void extractEyebrowsPoints();
    void preprocessEyebrowROI(Mat& src, Mat& dst, ROItype roi);
    void findEyebrowPoints(Mat& src, ROItype roi);

    void extractMouthPoints();
    void preprocessMouthROI(Mat& src, Mat& dst);
    void findMouthPoints(Mat& src);

    void extractTeethParam();
    void extractNosePoints();

    void collectFacialFeatures();

public:
    Mat faceFrame, faceFrameVis; // copy for displaying
    Point roiOffsets[OFF_NUM];
    vector<Point>* featureContours;
};

#endif // FacialFeatures_H
