#ifndef UTILS_H
#define UTILS_H

#include "Headers.h"

class Utils
{

    public:
        Utils();

        // Benchmarking
        double clock();
        double avgdur(double newdur);
        double avgfps();

    private:
        double _avgdur = 0;
        double _fpsstart = 0;
        double _avgfps = 0;
        double _fps1sec = 0;
};

#endif // UTILS_H
