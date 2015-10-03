#ifndef LOGGER_H
#define LOGGER_H

#include "Headers.h"

class Logger
{

public:
    Logger();
    ~Logger();
    void show(const double* fv, bool toFile = false);
    void showHeader();
    void showFeatureVector(const double* fv, bool toFile = false);

private:
    void cls();
};

#endif // LOGGER_H
