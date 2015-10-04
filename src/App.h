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
    void runImage(string imgPath, bool toFile = false, string subDir = Logger::getTime(), string outId = "");

private:
    Logger* log;
    FacialFeatures* facialFeatures;

    map<string, string> pathMap;
    void ensureDirectories(map<string, string> pathMap);
    void ensureDirectory(string path, bool inout = false);

};

#endif // APP_H
