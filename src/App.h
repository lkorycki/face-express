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
    void runImage(string imgPath, bool toFile = false);
    void captureSequence(int camId, int frames, int fps, bool features);

private:
    Logger* log;
    FacialFeatures* facialFeatures;

    map<string, string> pathMap;
    void ensureDirectories(map<string, string> pathMap);


};

#endif // APP_H
