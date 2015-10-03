#include "MathCore.h"

MathCore::MathCore()
{
}

double MathCore::dist2D(Point p1, Point p2)
{
    return sqrt((p1.x-p2.x)*(p1.x-p2.x) + (p1.y-p2.y)*(p1.y-p2.y));
}

Point MathCore::center2D(Rect rect)
{
    return Point(rect.width/2, rect.height/2);
}

float MathCore::avg2D(Mat& mat, int channel)
{
    float sum = 0;

    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            Vec3b& pixel = mat.at<Vec3b>(Point(j,i));

            sum += pixel[channel];
        }
    }

    return sum/(mat.rows*mat.cols);
}

float MathCore::stdDeviation2D(Mat& mat, int channel)
{
    float avg = MathCore::avg2D(mat, channel);
    float sum = 0;

    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            uchar& pixel = mat.at<uchar>(Point(j,i));

            sum += (pixel-avg)*(pixel-avg);
        }
    }

    return sqrt(sum/(mat.rows*mat.cols));
}

int MathCore::histThresh2D(Mat& mat, float p)
{
    // Initialize parameters
    int histSize = 256;    // bin size
    float range[] = { 0, 255 };
    const float *ranges[] = { range };

    // Calculate histogram
    MatND hist;
    calcHist( &mat, 1, 0, Mat(), hist, 1, &histSize, ranges, true, false );

    int thresh = p*mat.rows*mat.cols;
    int sum = 0;
    int threshVal = 0;

    for( int h = 0; h < histSize; h++ )
    {
         float binVal = hist.at<float>(h);
         sum += binVal;

         if(sum < thresh) threshVal = h;
         else break;
    }

    return threshVal;
}

float MathCore::wbParam2D(Mat& mat)
{
    float bp = 0, wp = 0;

    for(int i = 0; i < mat.rows; i++)
    {
        for(int j = 0; j < mat.cols; j++)
        {
            uchar& pixel = mat.at<uchar>(Point(j,i));
            wp += pixel;
        }
    }

    wp /= 255; // num of white pixels
    bp = mat.rows*mat.cols - wp;

    return (float)(wp/bp);
}

