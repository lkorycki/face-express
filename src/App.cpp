#include "App.h"

App::App()
{
    // Init windows
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

    // Main inits
    this->log = new Logger();
    this->facialFeatures = new FacialFeatures();
}

void App::runCam(int camId)
{
    // Inits
    VideoCapture cap(camId); // open the default camera
    if(!cap.isOpened()) return; // check if we succeeded

    while(1)
    {
        Mat frame; cap >> frame;
        resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
        //imshow("Video", frame);

        Mat faceFrame = Mat();
        this->facialFeatures->detectFace(frame, faceFrame);
        double* featureVector = this->facialFeatures->extractFacialFeatures(faceFrame);

        this->log->show(featureVector);

        if(waitKey(1) >= 0) break;
    }
}

void App::runImage(string imgPath)
{
    Mat frame = imread(imgPath, CV_LOAD_IMAGE_COLOR);
    resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);

    Mat faceFrame = Mat();
    facialFeatures->detectFace(frame, faceFrame);
    double* featureVector = facialFeatures->extractFacialFeatures(faceFrame);

    if(!faceFrame.empty()) log->show(featureVector);
}

App::~App()
{
    delete log;
    delete facialFeatures;
}
