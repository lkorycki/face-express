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

    // App paths
    pathMap["app"] = "/tmp/face-express/";
    pathMap["outputs"] = pathMap["app"] + "outputs/";
    ensureDirectories(pathMap);
}

void App::runCam(int camId)
{
    // Inits
    VideoCapture cap(camId); // open the default camera
    if(!cap.isOpened()) return; // check if we succeeded
    string outDir = pathMap["outputs"] + "/" + Logger::getTime() + "/"; // for capturing
    int n = 0;

    char k;
    while(1)
    {
        Mat frame; cap >> frame;
        resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
        //imshow("Video", frame);

        Mat faceFrame = Mat();
        this->facialFeatures->detectFace(frame, faceFrame);
        double* featureVector = this->facialFeatures->extractFacialFeatures(faceFrame);

        this->log->show(featureVector);

        k = waitKey(1);
        if(k == ' ') // capture the actual result
        {
            ensureDirectory(outDir, true);
            stringstream ss; ss << (n++); string id = ss.str();

            imwrite(outDir + "in/" + id + "_input" + ".png", frame);
            log->writeToFile(featureVector, outDir + "out/" + id + "_vec");
            imwrite(outDir + "out/" + id + "_feat" + ".png", this->facialFeatures->faceFrameVis);
        }
        else if(k >= 0) break;
    }
}

void App::runImage(string imgPath, bool toFile, string subDir, string outId)
{
    Mat frame = imread(imgPath, CV_LOAD_IMAGE_COLOR);
    resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
    //imshow("FaceDet", frame);

    Mat faceFrame = Mat();
    this->facialFeatures->detectFace(frame, faceFrame);
    double* featureVector = this->facialFeatures->extractFacialFeatures(faceFrame);

    if(!faceFrame.empty()) log->show(featureVector);
    if(toFile)
    {
        // Ensure directories
        string outDir = pathMap["outputs"] + "/" + subDir + "/";
        ensureDirectory(outDir, true);

        // Write outputs
        //imwrite(outDir + "in/" + outId + "_input_face" + ".png", this->facialFeatures->faceFrame);
        log->writeToFile(featureVector, outDir + "out/" + outId + "_vec");
        imwrite(outDir + "out/" + outId + "_feat" + ".png", this->facialFeatures->faceFrameVis);
    }

    waitKey(100); // needed for event loop processing (highgui)
    cin.get();
}

void App::ensureDirectories(map<string, string> pathMap)
{
    cout << "Ensuring directories...\n";
    for(auto const &e : pathMap)
    {
        cout << "\tCreating directory: " + e.second << endl;
        ensureDirectory(e.second);
    }
}

void App::ensureDirectory(string path, bool inout)
{
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(dir);

    if(inout)
    {
        boost::filesystem::path dirIn(path + "/in/");
        boost::filesystem::create_directory(dirIn);
        boost::filesystem::path dirOut(path + "/out/");
        boost::filesystem::create_directory(dirOut);
    }
}

App::~App()
{
    delete log;
    delete facialFeatures;
}