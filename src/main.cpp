#include "Headers.h"

int main(int argc, char *argv[])
{
    FaceFeatures* faceFeatures = new FaceFeatures();

    cout << CV_VERSION << endl;
    VideoCapture cap(0); // open the default camera
    if(!cap.isOpened()) return 1; // check if we succeeded
    cout << "Camera detected!\n" << endl;

    namedWindow("FaceDet", WINDOW_NORMAL);
    moveWindow("FaceDet", 0,0);
    namedWindow("FaceFeature", WINDOW_NORMAL);
    moveWindow("FaceFeature", 300,0);

    namedWindow("work1", WINDOW_NORMAL);
    moveWindow("work1", 0,300);
    namedWindow("work2", WINDOW_NORMAL);
    moveWindow("work2", 300,300);
    namedWindow("work3", WINDOW_NORMAL);
    moveWindow("work3", 600,300);
    namedWindow("work4", WINDOW_NORMAL);
    moveWindow("work4", 900,300);

    while(1)
    {
        Mat frame; cap >> frame;
        resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
        //imshow("Video", frame);

        Mat faceFrame = Mat();
        faceFeatures->detectFace(frame, faceFrame);
        faceFeatures->extractFaceFeatures(faceFrame);

        if(waitKey(30) >= 0) break;
    }

    delete faceFeatures;
    return 0;
}
