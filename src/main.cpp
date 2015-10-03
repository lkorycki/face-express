#include "Headers.h"
#include "App.h"

int main(int argc, char *argv[])
{
    App app = App();
    //app.runCam(0);
    app.captureSequence(0, 5, 30, false);

    return 0;
}


