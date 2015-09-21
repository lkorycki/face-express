#include "FaceDetector.h"

FaceDetector::FaceDetector(string dataPath)
{
    this->face_cascade = new CascadeClassifier();
    this->face_cascade->load(dataPath);
    cout << "Cascade classifier ready!\n" << endl;
}

Mat FaceDetector::detectFace(Mat frame)
{
    Mat faceFrame = frame.clone();

    Mat gray;
    cvtColor(frame, gray, CV_BGR2GRAY);

    // Find faces
    vector< Rect_<int> > faces;
    this->face_cascade->detectMultiScale(gray, faces);

    Rect faceROI;
    double maxArea = 0;

    for(int i = 0; i < faces.size(); i++)
    {
        Rect ROI = faces[i];

        double area = ROI.area();
        if(area > maxArea)
        {
            maxArea = area;
            faceROI = ROI;
        }
    }

    if(maxArea != 0) faceFrame = faceFrame(faceROI);

    Mat f = frame.clone();
    rectangle(f, faceROI, CV_RGB(0, 255, 0), 2);
    imshow("FaceDet", f);

    return faceFrame;
}

FaceDetector::~FaceDetector()
{
    delete face_cascade;
}
