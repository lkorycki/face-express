#include "FacialFeatures.h"

FacialFeatures::FacialFeatures()
{
   // ROI and contours
   for(int i = 0; i < ROI_NUM; i++)  this->ROI[i] = Mat();
   for(int i = 0; i < OFF_NUM; i++) this->roiOffsets[i] = Point(0,0); // offsets
   this->featureContours = new vector<Point>[6];

    // Feature vectors
   this->featurePoints = new Point[FEAT_POINTS];
   for(int i = 0; i < FEAT_POINTS; i++) this->featurePoints[i] = Point(0,0);
   this->featureVector = new float[FEAT_NUM];
   for(int i = 0; i < FEAT_NUM; i++) this->featureVector[i] = -1;

   // Their offsets (see file: feature_desc)
   this->featPointOffsets[L_EYE] = 0;
   this->featPointOffsets[R_EYE] = 5;
   this->featPointOffsets[L_EB] = 10;
   this->featPointOffsets[R_EB] = 13;
   this->featPointOffsets[MOUTH] = 16;
   this->featPointOffsets[NOSE] = 20;

   // ROI for processing and feature points visualization
   this->faceFrame = Mat();
   this->faceFrameVis = Mat();

   // Support modules set and read the facial features arrays
   ImageAnalyzer::setFF(this);
   ImageProcessor::setFF(this);
}

void FacialFeatures::detectFace(Mat& src, Mat& dst)
{
    // Detect with cascade classifier
    Rect faceROI = Rect();
    Mat f = src.clone();
    ImageAnalyzer::findBestObject(src, faceROI, "../data/haarcascade_frontalface_alt2.xml");

    if(faceROI.area() > 0) // found
    {
        ROI[FACE] = dst = src(faceROI);
        rectangle(f, faceROI, CV_RGB(0, 255, 0), 2);
    }

    imshow("FaceDet", f);
}

float* FacialFeatures::extractFacialFeatures(Mat& src)
{
    if(src.empty()) // no face detected
    {
        imshow("FaceFeature", NULL);
        for(int i = 0; i < FEAT_NUM; i++) this->featureVector[i] = -1;
        return this->featureVector;
    }

    // Save ROI
    this->faceFrame = src.clone();
    this->faceFrameVis = src.clone();

    // Extract facial features points
    setROI(src); // set ROIs for analyzing
    extractEyesPoints();
    extractEyebrowsPoints();
    extractMouthPoints();
    extractTeethParam();
    extractNosePoints();
    imshow("FaceFeature", this->faceFrameVis);

    // Collect (parametrize) facial features
    collectFacialFeatures();
    return this->featureVector;
}

void FacialFeatures::setROI(Mat& src)
{
    float fw = src.cols;
    float fh = src.rows;

    float cx = fw/2;
    float cy = fh/2;
    //circle(this->faceFrameVis, Point(cx, cy), 1, Scalar(0,255,255), 3);

    // 1) Eyes ROI
    float a = 0.35, b = 0.19; // based on [1] a = 0.4 b = 0.1666
    this->roiOffsets[L_EYE] = Point(cx-a*fw, cy-b*fh); this->roiOffsets[R_EYE] = Point(cx, cy-b*fh); // displaying offsets
    Rect leftEye = Rect(this->roiOffsets[L_EYE].x, this->roiOffsets[L_EYE].y, a*fw, b*fh);
    Rect rightEye = Rect(this->roiOffsets[R_EYE].x, this->roiOffsets[R_EYE].y, a*fw, b*fh);

    ROI[L_EYE] = src(leftEye);
    ROI[R_EYE] = src(rightEye);
    //rectangle(this->faceFrameVis, leftEye, CV_RGB(0, 0, 255), 1);
    //rectangle(this->faceFrameVis, rightEye, CV_RGB(0, 0, 255), 1);

    // 2) Eyebrows ROI
    a = 0.38, b = 0.22; // based on [1] a = 0.43 b = 0.2
    int ncy = cy - 0.1*fh;
    this->roiOffsets[L_EB] = Point(cx-a*fw, ncy-b*fh); this->roiOffsets[R_EB] = Point(cx, ncy-b*fh); // displaying offset
    Rect leftEyeBrow = Rect(this->roiOffsets[L_EB].x, this->roiOffsets[L_EB].y, a*fw, b*fh);
    Rect rightEyeBrow = Rect(this->roiOffsets[R_EB].x, this->roiOffsets[R_EB].y, a*fw, b*fh);

    ROI[L_EB] = src(leftEyeBrow);
    ROI[R_EB] = src(rightEyeBrow);
    //rectangle(this->faceFrameVis, leftEyeBrow, CV_RGB(0, 200, 0), 1);
    //rectangle(this->faceFrameVis, rightEyeBrow, CV_RGB(0, 200, 0), 1);

    // 3) Mouth ROI
    a *= 0.75, b = 0.1;
    this->roiOffsets[MOUTH] = Point(cx-a*fw, cy+b*fh);
    Rect mouth = Rect(roiOffsets[MOUTH].x, roiOffsets[MOUTH].y, 2*a*fw, 4*b*fh);

    ROI[MOUTH] = src(mouth);
    //rectangle(this->faceFrameVis, mouth, CV_RGB(255, 0, 0), 1);

    // 4) Nose ROI
    a = 0.3, b = 0.3;
    this->roiOffsets[NOSE] = Point(cx-a*fw, cy);
    Rect nose = Rect(this->roiOffsets[NOSE].x, this->roiOffsets[NOSE].y, 2*a*fw, b*fh);

    ROI[NOSE] = src(nose);
    //rectangle(this->faceFrameVis, nose, CV_RGB(0, 255, 255), 1);
}

void FacialFeatures::extractEyesPoints()
{
    Mat leftEye = Mat(ROI[L_EYE].rows, ROI[L_EYE].cols, CV_8U);
    Mat rightEye = Mat(ROI[R_EYE].rows, ROI[R_EYE].cols, CV_8U);

    // Preprocess eye ROI
    ImageProcessor::preprocessEyeROI(ROI[L_EYE], leftEye);
    ImageProcessor::preprocessEyeROI(ROI[R_EYE], rightEye);

    // Find features
    ImageAnalyzer::findEyePoints(leftEye, L_EYE);
    ImageAnalyzer::findEyePoints(rightEye, R_EYE);

    //imshow("work1", leftEye);
    //imshow("work2", rightEye);
    //imshow("LeftEye", ROI[L_EYE]);
    //imshow("RightEye", ROI[R_EYE]);
}

void::FacialFeatures::extractEyebrowsPoints()
{
    Mat leftEyebrow = Mat(ROI[L_EB].rows, ROI[L_EB].cols, CV_8U);
    Mat rightEyebrow = Mat(ROI[R_EB].rows, ROI[R_EB].cols, CV_8U);

    // Preprocess eyebrow ROI
    ImageProcessor::preprocessEyebrowROI(ROI[L_EB], leftEyebrow, L_EB);
    ImageProcessor::preprocessEyebrowROI(ROI[R_EB], rightEyebrow, R_EB);

    // Find features
    ImageAnalyzer::findEyebrowPoints(leftEyebrow, L_EB);
    ImageAnalyzer::findEyebrowPoints(rightEyebrow, R_EB);

    //imshow("work1", leftEyebrow);
    //imshow("work2", rightEyebrow);
    //imshow("Left", ROI[L_EB]);
    //imshow("Right", ROI[R_EB]);
}

void FacialFeatures::extractMouthPoints()
{
    Mat mouth = Mat(ROI[MOUTH].rows, ROI[MOUTH].cols, CV_8U);

    // Preprocess mouth ROI
    ImageProcessor::preprocessMouthROI(ROI[MOUTH], mouth);

    // Get features
    ImageAnalyzer::findMouthPoints(mouth);
}

void::FacialFeatures::extractTeethParam()
{
    if(this->featureContours[MOUTH].empty()) return; // no mouth detected

    // Get teeth ROI
    int off = 16; // detected mouth feature points -> dynamic teeth ROI
    Rect teethRect = Rect(this->featurePoints[off].x, this->featurePoints[off+1].y,
            this->featurePoints[off+2].x - this->featurePoints[off].x,
            this->featurePoints[off+3].y - this->featurePoints[off+1].y);

    if(!ImageAnalyzer::assertROI(this->faceFrame, teethRect)) return;
    ROI[TEETH] = this->faceFrame(teethRect);

    // Binarize
    Mat teethBin = Mat(ROI[TEETH].rows, ROI[TEETH].cols, CV_8U);
    ImageProcessor::binarizeTeeth(ROI[TEETH], teethBin, 75);

    // Get teeth param
    float wb = MathCore::wbParam2D(teethBin);
    this->featureVector[15] = wb;
    if(wb > 0.1) circle(this->faceFrameVis, Point(teethRect.x + teethRect.width/2, teethRect.y + teethRect.height/2),
                       3, Scalar(0,0,255), CV_FILLED);

    //imshow("work1", teethBin);
    //imshow("work2", ROI[TEETH]);
}

void::FacialFeatures::extractNosePoints()
{
    // Get nose ROI with cascade classifier
    int off = this->featPointOffsets[NOSE];
    Rect noseROI;
    ImageAnalyzer::findBestObject(ROI[NOSE], noseROI, "../data/nose_cascade.xml");
    if(!noseROI.area() && !this->featurePoints[off].x) return; // has not been found yet

    noseROI.x = noseROI.x + this->roiOffsets[NOSE].x; noseROI.y = noseROI.y + this->roiOffsets[NOSE].y; // equivalent of finding contours
    //rectangle(this->faceFrameVis, noseROI, CV_RGB(0, 0, 255), 1);

    // Get feature point
    this->featurePoints[off] = Point(noseROI.x + noseROI.width/2, noseROI.y + noseROI.height/2); // ~ nose tip
    line(this->faceFrameVis, Point(this->featurePoints[off].x, this->featurePoints[off].y-5),
         Point(this->featurePoints[off].x, this->featurePoints[off].y+5), Scalar(0,255,255));
    line(this->faceFrameVis, Point(this->featurePoints[off].x-5, this->featurePoints[off].y),
         Point(this->featurePoints[off].x+5, this->featurePoints[off].y), Scalar(0,255,255));
    circle(this->faceFrameVis, this->featurePoints[off], 2, Scalar(0,255,0), CV_FILLED);
    cout << "test3\n";
    //imshow("work1", this->faceFrame(noseROI));
}

void FacialFeatures::collectFacialFeatures()
{
    // Just for better coding
    Point* fp = this->featurePoints;
    float* fv = this->featureVector;

    // Features parametrization
    fv[0] = fp[1].x - fp[3].x; // left eye width
    fv[1] = fp[2].y - fp[0].y; // left eye height
    fv[2] = fp[6].x - fp[8].x; // right eye width
    fv[3] = fp[7].y - fp[5].y; // right eye height

    fv[4] = fp[4].y - fp[11].y; // left eye/eyebrow centers |dy|
    fv[5] = fp[3].y - fp[10].y; // outer left eye/eyebrow points |dy|
    fv[6] = fp[1].y - fp[12].y; // inner left eye/eyebrow points |dy|
    fv[7] = fp[9].y - fp[14].y; // right eye/eyebrow centers |dy|
    fv[8] = fp[6].y - fp[15].y; // outer right eye/eyebrow points |dy|
    fv[9] = fp[8].y - fp[13].y; // inner right eye/eyebrow points |dy|

    fv[10] = fp[18].x - fp[16].x; // mouth width
    fv[11] = fp[19].y - fp[17].y; // mouth height
    fv[12] = fp[19].y - fp[20].y; // lower lip/nose tip |dy|
    fv[13] = fp[16].y - fp[20].y; // left mouth corner/nose |dy|
    fv[14] = fp[18].y - fp[20].y; // right mouth corner/nose |dy|

    fv[15]; // teeth param (white pixels to black pixels)

    // Normalization
    float nf = fp[9].x - fp[4].x; // normalization factor is |dx| between eye centers
    for(int i = 0; i < FEAT_NUM-1; i++)
    {
        fv[i] /= nf;
        if(fv[i]  > 1.0) fv[i] = 1.0;
        else if(fv[i] < 0) fv[i] = 0;
    }
}

FacialFeatures::~FacialFeatures()
{
    delete[] this->featureContours;
    delete[] this->featurePoints;
    delete[] this->featureVector;
}

// -----------------------------------------
// [1] http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2010/09.08.18.08/doc/CameraReady_70662.pdf
