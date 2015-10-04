#include "Headers.h"
#include "App.h"

int main(int argc, char *argv[])
{
    App app = App();
    //app.runCam(0);
    app.runSequence(0, 10, 30, true);
    //app.runImage("/tmp/face-express/test.png", true);

    return 0;
}


