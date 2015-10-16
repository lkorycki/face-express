#ifndef LOGGER_H
#define LOGGER_H

#include "Headers.h"
#include "IntelliCore.h"

class Logger
{

public:
    Logger();
    ~Logger();
    void show(const float* fv, const float* ev, int e1, int e2, int e3);
    void showHeader();
    void showFeatureVector(const float* fv);
    void writeToFile(const float* fv, string path);
    void showEmotionSupport(const float* ev);
    void showEmotionRecognition(const float* ev, int e1, int e2, int e3);
    static string getTime();

private:
    void cls();

};

#endif // LOGGER_H
