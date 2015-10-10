#include "ImageProcessor.h"

FacialFeatures* ImageProcessor::ff;

ImageProcessor::ImageProcessor()
{
}

void ImageProcessor::setFF(FacialFeatures *ff)
{
    ImageProcessor::ff = ff;
}

void ImageProcessor::createEyeMap(Mat& src, Mat& dst)
{
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            Vec3b& pixel = src.at<Vec3b>(Point(j,i));
            uchar r = pixel[2];
            dst.at<uchar>(Point(j,i)) = exp((255-r)*(log(255.0)/255.0));
        }
    }
}

void ImageProcessor::binarizeEye(Mat& src, Mat& dst)
{
    float avg = MathCore::avg2D(src, 0);
    float dev = MathCore::stdDeviation2D(src, 0);

    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            uchar& src_pixel = src.at<uchar>(Point(j,i));
            uchar& dst_pixel = src.at<uchar>(Point(j,i));
            if(src_pixel > (avg + 0.9*dev)) dst_pixel = 255;
            else dst_pixel = 0;
        }
    }

    dilate(dst, dst, getStructuringElement( MORPH_RECT, Size(5,3)));

    // Clear border
    ImageProcessor::clearBinBorder(dst, dst);
}

void ImageProcessor::clearBinBorder(Mat& src, Mat& dst)
{
    dst = src.clone();

    // Top and bottom
    for(int y = 0; y < dst.rows; y += dst.rows-1)
    {
        uchar* row = dst.ptr<uchar>(y);
        for(int x = 0; x < dst.cols; ++x)
        {
            if(row[x] == 255) floodFill(dst, Point(x,y), Scalar(0));
        }
    }

    // Left and right
    for(int y = 0; y < dst.rows; ++y)
    {
        uchar* row = dst.ptr<uchar>(y);
        for(int x = 0; x < dst.cols; x += dst.cols - 1)
        {
            if(row[x] == 255) floodFill(dst, Point(x,y), Scalar(0));
        }
     }
}

void ImageProcessor::binarizeEyebrow(Mat& src, Mat& dst, float p, int ys)
{
    // Calculate thresh
    int threshVal = MathCore::histThresh2D(src, p);
    threshold(src, dst, threshVal, 255, THRESH_BINARY);

    // Mask detected eye
    if(ys <= -1) return;
    for(int i = ys; i < dst.rows; i++)
    {
        for(int j = 0; j < dst.cols; j++)
        {
           dst.at<uchar>(Point(j,i)) = 255;
        }
    }

    // Post processing
    medianBlur(dst, dst, 5);
    negateMat(dst, dst);
}

void ImageProcessor::clearGrayBorderV(Mat &src, Mat &dst, int lw, float lmax, int rw, float rmax)
{
    // penalty = -(max/width)*i + max (from left to right)

    // Left
    for(int i = 0; i < lw; i++)
    {
        for(int j = 0; j < src.rows; j++)
        {
            uchar& srcPixel = src.at<uchar>(Point(i,j));
            int newVal = (-(lmax/lw)*i + lmax)*(255-srcPixel) + srcPixel;
            if(newVal > 255) newVal = 255;
            dst.at<uchar>(Point(i,j)) = newVal;
        }
    }

    // Right
    for(int i = 0; i < rw; i++)
    {
        for(int j = 0; j < src.rows; j++)
        {
            uchar& srcPixel = src.at<uchar>(Point(src.cols-i-1,j));
            int newVal = (-(rmax/rw)*i + rmax)*(255-srcPixel) + srcPixel;
            if(newVal > 255) newVal = 255;
            dst.at<uchar>(Point(src.cols-i-1,j)) = newVal;
        }
    }
}

void ImageProcessor::clearGrayBorderH(Mat &src, Mat &dst, int tw, float tmax, int bw, float bmax)
{
    // Top
    for(int i = 0; i < tw; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            uchar& srcPixel = src.at<uchar>(Point(j,i));
            int newVal = (-(tmax/tw)*i + tmax)*(255-srcPixel) + srcPixel;
            if(newVal > 255) newVal = 255;
            dst.at<uchar>(Point(j,i)) = newVal;
        }
    }

    // Bottom
    for(int i = 0; i < bw; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            uchar& srcPixel = src.at<uchar>(Point(j,src.rows-i-1));
            int newVal = (-(bmax/bw)*i + bmax)*(255-srcPixel) + srcPixel;
            if(newVal > 255) newVal = 255;
            dst.at<uchar>(Point(j,src.rows-i-1)) = newVal;
        }
    }
}

void ImageProcessor::createMouthMap(Mat& src, Mat& dst)
{
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            Vec3b& pixel = src.at<Vec3b>(Point(j,i));
            float s1 = 255*2*atan((pixel[2]-pixel[1])/(float)pixel[2])/M_PI;
            dst.at<uchar>(Point(j,i)) = s1 < 0 ? 255 : 255-s1;
        }
    }

    equalizeHist(dst, dst);
}

void ImageProcessor::binarizeMouth(Mat& src, Mat& dst, float p)
{
    // Calculate thresh
    int threshVal = MathCore::histThresh2D(src, p);
    threshold(src, dst, threshVal, 255, THRESH_BINARY);

    // Postprocess
    //medianBlur(dst, dst,3); // ? (consider)
    erode(dst,dst, getStructuringElement( MORPH_ELLIPSE, Size(3,3)));
    negateMat(dst,dst);
}

void ImageProcessor::binarizeTeeth(Mat& src, Mat& dst, int t)
{
    // Convert to grayscale
    Mat gray; cvtColor(src, gray, CV_BGR2GRAY);
    Mat bin = Mat(gray.rows, gray.cols, CV_8U);

    // Preprocess
    /*
    equalizeHist(gray, gray);
    negateMat(gray, gray);
    */

    createMouthMap(src, gray); // ? (consider)
    negateMat(gray, gray);
    clearGrayBorderH(gray, gray, 0.35*gray.rows, 0.9, 0.4*gray.rows, 1.0);
    clearGrayBorderV(gray, gray, 0.15*gray.cols, 0.8, 0.15*gray.cols, 0.8);

    // Binarize
    threshold(gray, bin, t, 255, THRESH_BINARY);
    negateMat(bin, bin);

    dst = bin;
}

void ImageProcessor::negateMat(Mat& src, Mat& dst)
{
    for(int i = 0; i < src.rows; i++)
    {
        for(int j = 0; j < src.cols; j++)
        {
            uchar& pixel = src.at<uchar>(Point(j,i));
            dst.at<uchar>(Point(j,i)) = 255-pixel;
        }
    }
}

void ImageProcessor::findCorners(vector<Point> contour, Point& p1, Point& p2, bool horizontal)
{
    if(horizontal)
    {
        p1.x = INT_MAX; // left
        p2.x = -1; // right

        for(int i = 0; i < contour.size(); i++)
        {
            Point p = contour[i];

            if(p.x < p1.x) p1 = p;
            if(p.x > p2.x) p2 = p;
        }
    }
    else // vertical
    {
        p1.y = INT_MAX; // top
        p2.y = -1; // bot

        for(int i = 0; i < contour.size(); i++)
        {
            Point p = contour[i];

            if(p.y < p1.y) p1 = p;
            if(p.y > p2.y) p2 = p;
        }
    }
}

void ImageProcessor::createMouthCornerMap(Mat& src, Mat& dst, float p)
{
    Mat gray; cvtColor(src, gray, CV_BGR2GRAY);
    Mat map = Mat(src.rows, src.cols, CV_8U);

    for(int i = 0; i < gray.rows; i++)
    {
        for(int j = 0; j < gray.cols; j++)
        {
            uchar& pixel = gray.at<uchar>(Point(j,i));
            map.at<uchar>(Point(j,i)) = pixel;
        }
    }

    equalizeHist(map, map);
    ImageProcessor::clearGrayBorderH(map, map, 0, 0, 0.2*map.rows, 0.8);
    int thresh = MathCore::histThresh2D(map, p);
    threshold(map, dst, thresh, 255, THRESH_BINARY);
    negateMat(dst, dst);
}

void ImageProcessor::preprocessEyeROI(Mat& src, Mat& dst)
{
    // Create map of eye
    createEyeMap(src, dst);

    // Binarize eye
    binarizeEye(dst, dst);
}

void ImageProcessor::preprocessEyebrowROI(Mat& src, Mat& dst, ROIType roi)
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
    clearGrayBorderV(dst, dst, lw*dst.cols, lm, rw*dst.cols, rm);
    clearGrayBorderH(dst, dst, tw*dst.rows, tm, bw*dst.rows, bm);
    binarizeEyebrow(dst, dst, 0.15, ff->featurePoints[eyeOff].y-ff->roiOffsets[roi].y);
}

void ImageProcessor::preprocessMouthROI(Mat& src, Mat& dst)
{
    // Create mouth map
    createMouthMap(src, dst);
    clearGrayBorderH(dst, dst, 0.35*dst.rows, 1.0, 0.1*dst.rows, 1.0);

    // Binarize
    binarizeMouth(dst, dst, 0.1);
}

ImageProcessor::~ImageProcessor()
{
    ImageProcessor::ff = NULL;
}


