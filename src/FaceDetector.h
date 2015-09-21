#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "Headers.h"

class FaceDetector
{

public:
    FaceDetector(string dataPath);
    ~FaceDetector();
    Mat detectFace(Mat frame);

private:

    CascadeClassifier* face_cascade;
};

#endif // FACEDETECTOR_H
