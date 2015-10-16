#ifndef PROCORES_H
#define PROCORES_H

#include "Headers.h"

class MathCore
{

public:
    MathCore();
    static double dist2D(const Point& p1, const Point& p2);
    static Point center2D(const Rect& rect);
    static float avg2D(const Mat& mat, int channel);
    static float stdDeviation2D(const Mat& mat, int channel);
    static int histThresh2D(const Mat& mat, float p);
    static float wbParam2D(const Mat& mat);

    template<typename Type>
    static int maxPos(Type arr[], int size)
    {
        int maxIdx = 0;
        Type maxVal = arr[0];

        for(int i = 1; i < size; i++)
        {
            if(arr[i] > maxVal)
            {
                maxIdx = i;
                maxVal = arr[i];
            }
        }

        return maxIdx;
    }

};

#endif // PROCORES_H
