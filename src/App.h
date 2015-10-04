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
    void runImage(string imgPath, bool toFile = false, string outDir = "", string outId = Logger::getTime());
    void captureSequence(int camId, int frames, int delay, bool features);

private:
    Logger* log;
    FacialFeatures* facialFeatures;

    map<string, string> pathMap;
    void ensureDirectories(map<string, string> pathMap);
    void ensureDirectory(string path);
};

#endif // APP_H
