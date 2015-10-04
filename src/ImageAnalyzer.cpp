#include "ImageAnalyzer.h"

FacialFeatures* ImageAnalyzer::ff;

ImageAnalyzer::ImageAnalyzer()
{
}

void ImageAnalyzer::setFF(FacialFeatures *ff)
{
    ImageAnalyzer::ff = ff;
}

void ImageAnalyzer::findEyePoints(Mat& src, ROItype roi)
{
    // Find eye contour
    vector<Point> contour;
    findBestContour(src, contour, ff->roiOffsets[roi], roi);

    // Fit ellipse
    if(contour.size() >= 5)
    {
        ff->featureContours[roi] = contour;
    }
    else if(ff->featureContours[roi].empty()) return;

    // Get eye points
    RotatedRect elp = fitEllipse(ff->featureContours[roi]);
    Point2f points[4];
    elp.points(points);

    int off = ff->featPointOffsets[roi];
    if(roi == L_EYE) ff->featurePoints[4] = elp.center;
    else if(roi == R_EYE) ff->featurePoints[9] = elp.center;

    for(int i = 0; i < 4; i++)  ff->featurePoints[i+off] = Point((points[(i+1)%4].x + points[i].x)/2, (points[(i+1)%4].y + points[i].y)/2);
    for(int i = 0; i < 5; i++) circle(ff->faceFrameVis, ff->featurePoints[i+off], 2, Scalar(0,255,0), CV_FILLED);
    ellipse(ff->faceFrameVis, elp, Scalar(255,255,0));
}

void ImageAnalyzer::findEyebrowPoints(Mat &src, ROItype roi)
{
    // Find eyebrow contour
    vector<Point> contour;
    findBestContour(src, contour, ff->roiOffsets[roi]);
    if(!contour.empty()) ff->featureContours[roi] = contour;
    else if(ff->featureContours[roi].empty()) return;

    // Get eyebrow points
    int off = ff->featPointOffsets[roi];
    ImageProcessor::findCorners(ff->featureContours[roi], ff->featurePoints[off], ff->featurePoints[off+2]); // left, right
    Moments m = moments(ff->featureContours[roi], true);
    ff->featurePoints[off+1] = Point(m.m10/m.m00, m.m01/m.m00); // center
    line(ff->faceFrameVis, ff->featurePoints[off], ff->featurePoints[off+1], Scalar(0,255,255));
    line(ff->faceFrameVis, ff->featurePoints[off+1], ff->featurePoints[off+2], Scalar(0,255,255));
    for(int i = 0; i < 3; i++) circle(ff->faceFrameVis, ff->featurePoints[i+off], 2, Scalar(0,255,0), CV_FILLED);
}

void ImageAnalyzer::findMouthPoints(Mat& src)
{
    // Find mouth contour
    vector<Point> contour;
    findBestContour(src, contour, ff->roiOffsets[MOUTH]);

    // Fit ellipse
    RotatedRect elp;
    if(contour.size() >= 5) ff->featureContours[MOUTH] = contour;
    else if(ff->featureContours[MOUTH].size() < 5) return;

    elp = fitEllipse(ff->featureContours[MOUTH]);
    //ellipse(ff->faceFrameVis, elp, Scalar(255,0,120));

    // Find top and bot corner point
    int off = ff->featPointOffsets[MOUTH];
    Point2f points[4];
    elp.points(points);

    ff->featurePoints[off+1] = Point((points[0].x+points[1].x)/2, (points[0].y+points[1].y)/2); // top
    ff->featurePoints[off+3] = Point((points[2].x+points[3].x)/2, (points[2].y+points[3].y)/2); // bottom

    // Extract left and right corner ROI
    Point lc = Point((points[0].x+points[3].x)/2, (points[0].y+points[3].y)/2);
    Point rc = Point((points[1].x+points[2].x)/2, (points[1].y+points[2].y)/2);
    //circle(ff->faceFrameVis, lc, 3, Scalar(255,0,0), CV_FILLED);
    //circle(ff->faceFrameVis, rc, 3, Scalar(255,0,0), CV_FILLED);

    float a = 0.2;
    int fw = src.cols, fh = src.rows;
    Point leftOffset = Point(lc.x-a*fw, lc.y-a*fh); Point rightOffset = Point(rc.x-a*fw, rc.y-a*fh);
    Rect lr = Rect(leftOffset.x, leftOffset.y, 2*a*fw, 2*a*fh);
    Rect rr = Rect(rightOffset.x, rightOffset.y, 2*a*fw, 2*a*fh);
    if(lr.x + lr.width > ff->faceFrame.cols || rr.x + rr.width > ff->faceFrame.cols
            || lr.y + lr.height > ff->faceFrame.rows || rr.y + rr.height > ff->faceFrame.rows
            || lr.x < 0 || lr.y < 0 || rr.x < 0 || rr.y < 0) return;

    Mat leftROI = ff->faceFrame(lr);
    Mat rightROI = ff->faceFrame(rr);
    //rectangle(ff->faceFrameVis, lr, Scalar(0,0,255));
    //rectangle(ff->faceFrameVis, rr, Scalar(0,0,255));

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
            ff->featurePoints[off] = pLeft; // left
            ff->featurePoints[off+2] = pRight; // right
        }
    }
    else if(!ff->featurePoints[off].x || !ff->featurePoints[off+2].x) return;

    for(int i = 0; i < 4; i++)
    {
        line(ff->faceFrameVis, ff->featurePoints[i+off], ff->featurePoints[((i+1)%4)+off], Scalar(120,0,255));
    }
    circle(ff->faceFrameVis, ff->featurePoints[off+1], 2, Scalar(0,255,0), CV_FILLED); // top
    circle(ff->faceFrameVis, ff->featurePoints[off+3], 2, Scalar(0,255,0), CV_FILLED); // bot
    circle(ff->faceFrameVis, ff->featurePoints[off], 2, Scalar(0,255,0), CV_FILLED); // left
    circle(ff->faceFrameVis, ff->featurePoints[off+2], 2, Scalar(0,255,0), CV_FILLED); //right

    //imshow("Left", leftROI);
    //imshow("Right", rightROI);
}

void ImageAnalyzer::findBestContour(Mat& src, vector<Point>& contour, Point offset, ROItype roi)
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
        //drawContours(ff->faceFrame, contours, idx, Scalar(0,255,255), 1);
    }
}

void ImageAnalyzer::findBestObject(Mat& src, Rect& dstROI, string dataPath)
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

ImageAnalyzer::~ImageAnalyzer()
{
    ImageAnalyzer::ff = NULL;
}
