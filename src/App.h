#ifndef APP_H
#define APP_H

#include "Headers.h"
#include "FacialFeatures.h"
#include "Logger.h"

class App
{

public:
    App();
    ~App();
    void runCam(int camId);
    void runImage(string imgPath);

private:
    Logger* log;
    FacialFeatures* facialFeatures;

};

#endif // APP_H
