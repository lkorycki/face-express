#include "Headers.h"

void cls();

int main(int argc, char *argv[])
{
    // Inits
    Logger* log = new Logger();

    FacialFeatures* facialFeatures = new FacialFeatures();
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()) return 1; // check if we succeeded

    namedWindow("FaceDet", WINDOW_NORMAL);
    moveWindow("FaceDet", 0,0);
    namedWindow("FaceFeature", WINDOW_NORMAL);
    moveWindow("FaceFeature", 300,0);

//    namedWindow("work1", WINDOW_NORMAL);
//    moveWindow("work1", 0,300);
//    namedWindow("work2", WINDOW_NORMAL);
//    moveWindow("work2", 300,300);
//    namedWindow("work3", WINDOW_NORMAL);
//    moveWindow("work3", 600,300);
//    namedWindow("work4", WINDOW_NORMAL);
//    moveWindow("work4", 900,300);

    while(1)
    {
        Mat frame; cap >> frame;
        resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
        //imshow("Video", frame);

        Mat faceFrame = Mat();
        facialFeatures->detectFace(frame, faceFrame);
        double* featureVector = facialFeatures->extractFacialFeatures(faceFrame);

        log->show(featureVector);

        if(waitKey(1) >= 0) break;
    }

    delete log;
    delete facialFeatures;
    return 0;
}


