#ifndef INTELLICORE_H
#define INTELLICORE_H

#include "Headers.h"

class IntelliCore
{

public:
    IntelliCore();
    void loadCSV(string path, Mat& input, Mat& target, int N, int X);
    ~IntelliCore();

};

#endif // INTELLICORE_H
