#include "Headers.h"
#include "App.h"

int main(int argc, char *argv[])
{
    App app = App();
    //cout << argc << endl;

    app.runCam(0);
    //app.runImage("/tmp/face-express/test.png", true);

    return 0;
}


