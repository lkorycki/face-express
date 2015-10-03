#include "Headers.h"
#include "App.h"

int main(int argc, char *argv[])
{
    App app = App();
    //app.runCam(0);
    //app.captureSequence(0, 10, 30, false);
    app.runImage("/tmp/face-express/seq/7.png");

    return 0;
}


