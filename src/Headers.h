#ifndef HEADERS_H
#define HEADERS_H

#include <math.h>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
using namespace std;

#include <opencv2/opencv.hpp>
#include <opencv2/ml/ml.hpp>
using namespace cv;

#include <floatfann.h>
#include <fann_cpp.h>
using namespace FANN;

enum ROIType { L_EYE, R_EYE, L_EB, R_EB, MOUTH, NOSE, TEETH, FACE, NONE };
enum EmotionType { NEUTRAL, HAPPY, SURPRISE, ANGER, SAD, DISGUST, FEAR };

#endif // HEADERS_H
