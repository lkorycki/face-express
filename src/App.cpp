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

void App::runImage(string imgPath, bool toFile, string outDir, string outId)
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
        string dir = (outDir == "" ? pathMap["outputs"] : outDir);
        log->writeToFile(featureVector, dir + "vec_" + outId);
        imwrite(dir + "feat_" + outId + ".png", this->facialFeatures->faceFrameVis);
    }

    if(outDir == "")
    {
        waitKey(100); // needed for event loop processing (highgui)
        cin.get();
    }
}

void App::captureSequence(int camId, int frames, int delay, bool features)
{
    // Inits
    vector<Mat> seq;
    VideoCapture cap(camId); // open the default camera
    if(!cap.isOpened()) return; // check if we succeeded

    // Capture
    for(int i = 0; i < frames; i++)
    {
        Mat frame; cap >> frame;
        resize(frame, frame, Size(860, 640), 1.0, 1.0, INTER_CUBIC);
        imshow("FaceDet", frame);
        seq.push_back(frame);

        if(waitKey(delay) >= 0) break;
    }

    // Write from array of mats
    string dir = pathMap["outputs"] + log->getTime();
    ensureDirectory(dir);
    ensureDirectory(dir + "/in/");
    if(features) ensureDirectory(dir + "/out/");

    for(int i = 0; i < seq.size(); i++)
    {
        stringstream ss; ss << i;
        string path = dir + "/in/" + ss.str() + ".png";
        cout << "Writing to: " << path << endl;
        imwrite(path, seq[i]);

        // Extract features if needed
        if(features) runImage(path, true, dir + "/out/", ss.str());
    }
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

void App::ensureDirectory(string path)
{
    boost::filesystem::path dir(path);
    boost::filesystem::create_directory(path);
}

App::~App()
{
    delete log;
    delete facialFeatures;
}
