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

    // App paths
    pathMap["appCat"] = "/tmp/face-express";
    pathMap["seqCat"] = pathMap["appCat"] + "/seq";
    ensureDirectories(pathMap);

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
        // TODO: save mat and fv to file on click
    }
}

void App::runImage(string imgPath, bool toFile)
{
    Mat frame = imread(imgPath, CV_LOAD_IMAGE_COLOR);
    resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);

    Mat faceFrame = Mat();
    facialFeatures->detectFace(frame, faceFrame);
    double* featureVector = facialFeatures->extractFacialFeatures(faceFrame);

    if(!faceFrame.empty()) log->show(featureVector, toFile);
}

void App::captureSequence(int camId, int frames, int fps, bool features)
{
    // Inits
    vector<Mat> seq;
    //for(int i = 0; i < seq->size(); i++) seq[i] = Mat(860, 640, CV_8UC3);
    VideoCapture cap(camId); // open the default camera
    if(!cap.isOpened()) return; // check if we succeeded

    // Capture
    for(int i = 0; i < frames; i++)
    {
        Mat frame; cap >> frame;
        //cout << frame.type() << endl;
        resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
        imshow("FaceDet", frame);
        seq.push_back(frame);

        if(waitKey(fps) >= 0) break;
    }

    // Write from array of mats
    for(int i = 0; i < seq.size(); i++)
    {
        stringstream ss;
        ss << i;
        string path = pathMap["seqCat"] + "/" + ss.str() + ".png";
        cout << "Writing to: " << path << endl;
        imwrite(path, seq[i]);
    }

    // if(features) runImage for each

    //delete seq;
}

void App::ensureDirectories(map<string, string> pathMap)
{
    cout << "Ensuring directories...\n";
    for(auto const &e : pathMap)
    {
        //cout << "Creating directory: " + e.second << endl;
        boost::filesystem::path dir(e.second);
        boost::filesystem::create_directory(dir);
    }
}

App::~App()
{
    delete log;
    delete facialFeatures;
}
