#ifndef LOGGER_H
#define LOGGER_H

#include "Headers.h"

class Logger
{

public:
    Logger();
    ~Logger();
    void show(const double* fv);
    void showHeader();
    void showFeatureVector(const double* fv);

private:
    void cls();
};

#endif // LOGGER_H
