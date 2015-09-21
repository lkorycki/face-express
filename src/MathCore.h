#ifndef PROCORES_H
#define PROCORES_H

#include "Headers.h"

class MathCore
{

public:
    MathCore();
    static double dist2D(Point p1, Point p2);
    static Point center2D(Rect rect);
    static float avg2D(Mat& mat, int channel);
    static float stdDeviation2D(Mat& mat, int channel);
    static int histThresh2D(Mat& mat, float p);
    static float wbParam2D(Mat& mat);

};

#endif // PROCORES_H
