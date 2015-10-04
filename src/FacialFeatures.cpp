// [1] RCA: http://www.cse.unsw.edu.au/~tatjana/ICMLWS02/MLCV/Morales.pdf
// [2] http://sibgrapi.sid.inpe.br/col/sid.inpe.br/sibgrapi/2010/09.08.18.08/doc/CameraReady_70662.pdf

#include "FacialFeatures.h"

FacialFeatures::FacialFeatures()
{
   // ROI and contours
   for(int i = 0; i < ROI_NUM; i++)  this->ROI[i] = Mat();
   for(int i = 0; i < OFF_NUM; i++) this->roiOffsets[i] = Point(0,0); // offsets
   this->featureContours = new vector<Point>[6]; // todo

    // Feature vectors
   this->featurePoints = new Point[FEAT_POINTS];
   for(int i = 0; i < FEAT_POINTS; i++) this->featurePoints[i] = Point(0,0);
   this->featureVector = new double[FEAT_NUM];
   for(int i = 0; i < FEAT_NUM; i++) this->featureVector[i] = -1;

   // Their offsets
   this->featPointOffsets[L_EYE] = 0; this->featVecOffsets[L_EYE] = 0;
   this->featPointOffsets[R_EYE] = 5; this->featVecOffsets[R_EYE] = 0;
   this->featPointOffsets[L_EB] = 10; this->featVecOffsets[L_EB] = 0;
   this->featPointOffsets[R_EB] = 13; this->featVecOffsets[R_EB] = 0;
   this->featPointOffsets[MOUTH] = 16; this->featVecOffsets[MOUTH] = 0;
   this->featPointOffsets[NOSE] = 20; this->featVecOffsets[NOSE] = 0;
}

void FacialFeatures::detectFace(Mat& src, Mat& dst)
{
    // Detect with cascade classifier
    Rect faceROI = Rect();
    Mat f = src.clone();
    findBestObject(src, faceROI, "../data/haarcascade_frontalface_alt2.xml");

    if(faceROI.area() > 0)
    {
        ROI[FACE] = dst = src(faceROI);
        rectangle(f, faceROI, CV_RGB(0, 255, 0), 2);
    }

    imshow("FaceDet", f);
}

double* FacialFeatures::extractFacialFeatures(Mat& src)
{
    if(src.empty()) // no face detected
    {
        imshow("FaceFeature", NULL);
        for(int i = 0; i < FEAT_NUM; i++) this->featureVector[i] = -1;
        return this->featureVector;
    }
    this->faceFrame = src.clone();
    this->faceFrameVis = src.clone();

    // Extract facial features points
    setROI(src);
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
    float a = 0.35, b = 0.19; // based on [2] a = 0.4 b = 0.1666
    this->roiOffsets[L_EYE] = Point(cx-a*fw, cy-b*fh); this->roiOffsets[R_EYE] = Point(cx, cy-b*fh); // displaying offsets
    Rect leftEye = Rect(this->roiOffsets[L_EYE].x, this->roiOffsets[L_EYE].y, a*fw, b*fh);
    Rect rightEye = Rect(this->roiOffsets[R_EYE].x, this->roiOffsets[R_EYE].y, a*fw, b*fh);

    ROI[L_EYE] = src(leftEye);
    ROI[R_EYE] = src(rightEye);
    //rectangle(this->faceFrameVis, leftEye, CV_RGB(0, 0, 255), 1);
    //rectangle(this->faceFrameVis, rightEye, CV_RGB(0, 0, 255), 1);

    // 2) Eyebrows ROI
    a = 0.38, b = 0.22; // based on [2] a = 0.43 b = 0.2
    int ncy = cy - b*fh/2;
    this->roiOffsets[L_EB] = Point(cx-a*fw, ncy-b*fh); this->roiOffsets[R_EB] = Point(cx, ncy-b*fh); // displaying offset
    Rect leftEyeBrow = Rect(this->roiOffsets[L_EB].x, this->roiOffsets[L_EB].y, a*fw, b*fh);
    Rect rightEyeBrow = Rect(this->roiOffsets[R_EB].x, this->roiOffsets[R_EB].y, a*fw, b*fh);

    ROI[L_EB] = src(leftEyeBrow);
    ROI[R_EB] = src(rightEyeBrow);
    //rectangle(this->faceFrameVis, leftEyeBrow, CV_RGB(0, 200, 0), 1);
    //rectangle(this->faceFrameVis, rightEyeBrow, CV_RGB(0, 200, 0), 1);

    // 3) Mouth ROI
    a *= 0.75, b = 0.8;
    this->roiOffsets[MOUTH] = Point(cx-a*fw, cy+0.1*fh);
    Rect mouth = Rect(roiOffsets[MOUTH].x, roiOffsets[MOUTH].y, 2*a*fw, b*0.5*fh);

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
    preprocessEyeROI(ROI[L_EYE], leftEye);
    preprocessEyeROI(ROI[R_EYE], rightEye);

    // Find features
    findEyePoints(leftEye, L_EYE);
    findEyePoints(rightEye, R_EYE);

    //imshow("work1", leftEye);
    //imshow("work2", rightEye);
    //imshow("LeftEye", ROI[L_EYE]);
    //imshow("RightEye", ROI[R_EYE]);
}

void FacialFeatures::preprocessEyeROI(Mat& src, Mat& dst)
{
    // Create map of eye
    ImageProcessor::createEyeMap(src, dst);

    // Binarize eye
    ImageProcessor::binarizeEye(dst, dst);
}

void FacialFeatures::findEyePoints(Mat& src, ROItype roi)
{
    // Find eye contour
    vector<Point> contour;
    findBestContour(src, contour, this->roiOffsets[roi], roi);

    // Fit ellipse
    if(contour.size() >= 5)
    {
        this->featureContours[roi] = contour;
    }
    else if(this->featureContours[roi].empty()) return;

    // Get eye points
    RotatedRect elp = fitEllipse(this->featureContours[roi]);
    Point2f points[4];
    elp.points(points);

    int off = this->featPointOffsets[roi];
    if(roi == L_EYE) this->featurePoints[4] = elp.center;
    else if(roi == R_EYE) this->featurePoints[9] = elp.center;

    for(int i = 0; i < 4; i++)  this->featurePoints[i+off] = Point((points[(i+1)%4].x + points[i].x)/2, (points[(i+1)%4].y + points[i].y)/2);
    for(int i = 0; i < 5; i++) circle(this->faceFrameVis, this->featurePoints[i+off], 2, Scalar(0,255,0), CV_FILLED);
    ellipse(this->faceFrameVis, elp, Scalar(255,255,0));

    // Get eye features
    //this->featureVector[0] = elp_points[0]; // eye center
    this->featureVector[0] = elp.boundingRect().width; // eye width
    this->featureVector[1] = elp.boundingRect().height; // eye height
    this->featureVector[2] = elp.angle; // rotation angle
}

void::FacialFeatures::extractEyebrowsPoints()
{
    Mat leftEyebrow = Mat(ROI[L_EB].rows, ROI[L_EB].cols, CV_8U);
    Mat rightEyebrow = Mat(ROI[R_EB].rows, ROI[R_EB].cols, CV_8U);

    // Preprocess eyebrow ROI
    preprocessEyebrowROI(ROI[L_EB], leftEyebrow, L_EB);
    preprocessEyebrowROI(ROI[R_EB], rightEyebrow, R_EB);

    // Find features
    findEyebrowPoints(leftEyebrow, L_EB);
    findEyebrowPoints(rightEyebrow, R_EB);

    //imshow("work1", leftEyebrow);
    //imshow("work2", rightEyebrow);
    //imshow("Left", ROI[L_EB]);
    //imshow("Right", ROI[R_EB]);
}

void FacialFeatures::preprocessEyebrowROI(Mat& src, Mat& dst, ROItype roi)
{
    // Grayscale contrast
    cvtColor(src, dst, CV_BGR2GRAY);
    equalizeHist(dst, dst);

    // Binarize eyebrows
    int eyeOff;
    float lw, lm, rw, rm, tw, tm, bw, bm;

    tw = 0.15; tm = 0.75; bw = 0.2; bm = 0.75;
    if(roi == L_EB)
    {
        eyeOff = 0;
        lw = 0.25; lm = 0.85; rw = 0.1; rm = 0.6;
    }
    else
    {
        eyeOff = 5;
        lw = 0.1; lm = 0.6; rw = 0.25; rm = 0.8;
    }

    // Mask gray borders and binarize
    ImageProcessor::clearGrayBorderV(dst, dst, lw*dst.cols, lm, rw*dst.cols, rm);
    ImageProcessor::clearGrayBorderH(dst, dst, tw*dst.rows, tm, bw*dst.rows, bm);
    ImageProcessor::binarizeEyebrow(dst, dst, 0.15, this->featurePoints[eyeOff].y-this->roiOffsets[roi].y);
}

void FacialFeatures::findEyebrowPoints(Mat &src, ROItype roi)
{
    // Find eyebrow contour
    vector<Point> contour;
    findBestContour(src, contour, this->roiOffsets[roi]);
    if(!contour.empty()) this->featureContours[roi] = contour;
    else if(this->featureContours[roi].empty()) return;

    // Get eyebrow points
    int off = this->featPointOffsets[roi];
    ImageProcessor::findCorners(this->featureContours[roi], this->featurePoints[off], this->featurePoints[off+2]); // left, right
    Moments m = moments(this->featureContours[roi], true);
    this->featurePoints[off+1] = Point(m.m10/m.m00, m.m01/m.m00); // center
    line(this->faceFrameVis, this->featurePoints[off], this->featurePoints[off+1], Scalar(0,255,255));
    line(this->faceFrameVis, this->featurePoints[off+1], this->featurePoints[off+2], Scalar(0,255,255));
    for(int i = 0; i < 3; i++) circle(this->faceFrameVis, this->featurePoints[i+off], 2, Scalar(0,255,0), CV_FILLED);   
}

void FacialFeatures::extractMouthPoints()
{
    Mat mouth = Mat(ROI[MOUTH].rows, ROI[MOUTH].cols, CV_8U);

    // Preprocess mouth ROI
    preprocessMouthROI(ROI[MOUTH], mouth);

    // Get features
    findMouthPoints(mouth);
}

void FacialFeatures::preprocessMouthROI(Mat& src, Mat& dst)
{
    // Create mouth map
    ImageProcessor::createMouthMap(src, dst);
    ImageProcessor::clearGrayBorderH(dst, dst, 0.35*dst.rows, 1.0, 0.1*dst.rows, 1.0);

    // Binarize
    ImageProcessor::binarizeMouth(dst, dst, 0.1);
}

void FacialFeatures::findMouthPoints(Mat& src)
{
    // Find mouth contour
    vector<Point> contour;
    findBestContour(src, contour, this->roiOffsets[MOUTH]);

    // Fit ellipse
    RotatedRect elp;
    if(contour.size() >= 5) this->featureContours[MOUTH] = contour;
    else if(this->featureContours[MOUTH].size() < 5) return;

    elp = fitEllipse(this->featureContours[MOUTH]);
    //ellipse(this->faceFrameVis, elp, Scalar(255,0,120));

    // Find top and bot corner point
    int off = this->featPointOffsets[MOUTH];
    Point2f points[4];
    elp.points(points);

    this->featurePoints[off+1] = Point((points[0].x+points[1].x)/2, (points[0].y+points[1].y)/2); // top
    this->featurePoints[off+3] = Point((points[2].x+points[3].x)/2, (points[2].y+points[3].y)/2); // bottom

    // Extract left and right corner ROI
    Point lc = Point((points[0].x+points[3].x)/2, (points[0].y+points[3].y)/2);
    Point rc = Point((points[1].x+points[2].x)/2, (points[1].y+points[2].y)/2);
    //circle(this->faceFrameVis, lc, 3, Scalar(255,0,0), CV_FILLED);
    //circle(this->faceFrameVis, rc, 3, Scalar(255,0,0), CV_FILLED);

    float a = 0.2;
    int fw = src.cols, fh = src.rows;
    Point leftOffset = Point(lc.x-a*fw, lc.y-a*fh); Point rightOffset = Point(rc.x-a*fw, rc.y-a*fh);
    Rect lr = Rect(leftOffset.x, leftOffset.y, 2*a*fw, 2*a*fh);
    Rect rr = Rect(rightOffset.x, rightOffset.y, 2*a*fw, 2*a*fh);
    if(lr.x + lr.width > this->faceFrame.cols || rr.x + rr.width > this->faceFrame.cols
            || lr.y + lr.height > this->faceFrame.rows || rr.y + rr.height > this->faceFrame.rows
            || lr.x < 0 || lr.y < 0 || rr.x < 0 || rr.y < 0) return;

    Mat leftROI = this->faceFrame(lr);
    Mat rightROI = this->faceFrame(rr);
    //rectangle(this->faceFrameVis, lr, Scalar(0,0,255));
    //rectangle(this->faceFrameVis, rr, Scalar(0,0,255));

    // Create mouth corners binary map
    ImageProcessor::createMouthCornerMap(leftROI, leftROI, 0.05);
    ImageProcessor::createMouthCornerMap(rightROI, rightROI, 0.05);

    // Find their contours
    vector<Point> leftCorner, rightCorner;
    findBestContour(leftROI, leftCorner, leftOffset);
    findBestContour(rightROI, rightCorner, rightOffset);

    // Get left and right corner point
    if(!leftCorner.empty() && !rightCorner.empty())
    {
        Point p0, pLeft, pRight;
        ImageProcessor::findCorners(leftCorner, pLeft, p0);
        ImageProcessor::findCorners(rightCorner, p0, pRight);

        if(pLeft.x != INT_MAX && pRight.x != -1)
        {
            this->featurePoints[off] = pLeft; // left
            this->featurePoints[off+2] = pRight; // right
        }
    }
    else if(!this->featurePoints[off].x || !this->featurePoints[off+2].x) return;

    for(int i = 0; i < 4; i++)
    {
        line(this->faceFrameVis, this->featurePoints[i+off], this->featurePoints[((i+1)%4)+off], Scalar(120,0,255));
    }
    circle(this->faceFrameVis, this->featurePoints[off+1], 2, Scalar(0,255,0), CV_FILLED); // top
    circle(this->faceFrameVis, this->featurePoints[off+3], 2, Scalar(0,255,0), CV_FILLED); // bot
    circle(this->faceFrameVis, this->featurePoints[off], 2, Scalar(0,255,0), CV_FILLED); // left
    circle(this->faceFrameVis, this->featurePoints[off+2], 2, Scalar(0,255,0), CV_FILLED); //right

    //imshow("Left", leftROI);
    //imshow("Right", rightROI);
}

void::FacialFeatures::extractTeethParam()
{
    if(this->featureContours[MOUTH].empty()) return; // no mouth detected

    // Get teeth ROI
    int off = 16; // mouth rect
    Rect teethRect = Rect(this->featurePoints[off].x, this->featurePoints[off+1].y,
            this->featurePoints[off+2].x - this->featurePoints[off].x,
            this->featurePoints[off+3].y - this->featurePoints[off+1].y);

    if(teethRect.x + teethRect.width > this->faceFrame.cols
            || teethRect.y + teethRect.height > this->faceFrame.rows ) return;
    ROI[TEETH] = this->faceFrame(teethRect);

    // Binarize
    Mat teethBin;
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
    // Get nose ROI
    int off = this->featPointOffsets[NOSE];
    Rect noseROI;
    findBestObject(ROI[NOSE], noseROI, "../data/nose_cascade.xml");

    if(!noseROI.area() && !this->featurePoints[off].x) return; // if not found

    noseROI.x = noseROI.x + this->roiOffsets[NOSE].x; noseROI.y = noseROI.y + this->roiOffsets[NOSE].y; // equivalent of finding contours
    //rectangle(this->faceFrameVis, noseROI, CV_RGB(0, 0, 255), 1);

    // Get feature point
    this->featurePoints[off] = Point(noseROI.x + noseROI.width/2, noseROI.y + noseROI.height/2); // ~ nose tip
    line(this->faceFrameVis, Point(this->featurePoints[off].x, this->featurePoints[off].y-5),
         Point(this->featurePoints[off].x, this->featurePoints[off].y+5), Scalar(0,255,255));
    line(this->faceFrameVis, Point(this->featurePoints[off].x-5, this->featurePoints[off].y),
         Point(this->featurePoints[off].x+5, this->featurePoints[off].y), Scalar(0,255,255));
    circle(this->faceFrameVis, this->featurePoints[off], 2, Scalar(0,255,0), CV_FILLED);

    //imshow("work1", this->faceFrame(noseROI));
}

void FacialFeatures::findBestContour(Mat& src, vector<Point>& contour, Point offset, ROItype roi)
{
    // Find the biggest contour
    vector< vector<Point> > contours;
    int idx = -1;
    vector<Vec4i> hierarchy;
    double maxArea = -1;
    bool eye = false; if(roi == L_EYE || roi == R_EYE) eye = true;

    findContours(src.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, offset);
    for(int i = 0; i < contours.size(); i++)
    {
        double area = contourArea(contours[i]);
        if(!eye && area < maxArea) continue;
        else if(eye && (area < maxArea || boundingRect(contours[i]).width > 0.7*src.cols)) continue; // to distinguish eyebrows
        else
        {
            idx = i;
            maxArea = area;
        }
    }

    if(idx != -1)
    {
        contour = contours[idx];
        //drawContours(this->faceFrame, contours, idx, Scalar(0,255,255), 1);
    }
}

void FacialFeatures::findBestObject(Mat& src, Rect& dstROI, string dataPath)
{
    CascadeClassifier cascade = CascadeClassifier(dataPath); // init classifier
    Mat gray;
    cvtColor(src, gray, CV_BGR2GRAY);

    // Find objects
    vector< Rect_<int> > objects;
    cascade.detectMultiScale(gray, objects);

    Rect bestROI;
    double maxArea = 0;

    for(int i = 0; i < objects.size(); i++)
    {
        Rect ROI = objects[i];

        double area = ROI.area();
        if(area > maxArea)
        {
            maxArea = area;
            bestROI = ROI;
        }
    }

    if(maxArea != 0) dstROI = bestROI;
}

void FacialFeatures::collectFacialFeatures()
{
    // For better coding
    Point* fp = this->featurePoints;
    double* fv = this->featureVector;

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
    double nf = fp[9].x - fp[4].x; // normalization factor is |dx| between eye centers
    for(int i = 0; i < FEAT_NUM-1; i++) fv[i] /= nf;
}

FacialFeatures::~FacialFeatures()
{
    delete[] this->featureContours;
    delete[] this->featurePoints;
    delete[] this->featureVector;
}
