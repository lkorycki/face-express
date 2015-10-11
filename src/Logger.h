#ifndef LOGGER_H
#define LOGGER_H

#include "Headers.h"
#include "IntelliCore.h"

class Logger
{

public:
    Logger();
    ~Logger();
    void show(const double* fv, const double* ev);
    void showHeader();
    void showFeatureVector(const double* fv);   
    void writeToFile(const double* fv, string path);
    void showEmotionRecognition(const double* ev);
    static string getTime();

private:
    void cls();

};

#endif // LOGGER_H
