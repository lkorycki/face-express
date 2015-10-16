#ifndef FacialFeatures_H
#define FacialFeatures_H

#define ROI_NUM 8
#define OFF_NUM 6
#define FEAT_POINTS 21
#define FEAT_NUM 16

#include "Headers.h"
#include "MathCore.h"
#include "ImageProcessor.h"
#include "ImageAnalyzer.h"

class FacialFeatures
{

friend class ImageProcessor;
friend class ImageAnalyzer;

public:
    FacialFeatures();
    ~FacialFeatures();
    void detectFace(const Mat& src, Mat& dst);
    float* extractFacialFeatures(const Mat& src);
    void setROI(const Mat& src);

private:
    Mat ROI[ROI_NUM];
    Point* featurePoints;
    float* featureVector;
    int featPointOffsets[OFF_NUM];

    void extractEyesPoints();
    void extractEyebrowsPoints();
    void extractMouthPoints();
    void extractTeethParam();
    void extractNosePoints();

    void collectFacialFeatures();

public:
    Mat faceFrame, faceFrameVis; // copy for displaying
    Point roiOffsets[OFF_NUM];
    vector<Point>* featureContours;

};

#endif // FacialFeatures_H
