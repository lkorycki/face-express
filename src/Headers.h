#ifndef HEADERS_H
#define HEADERS_H

#include <math.h>
#include <fstream>
#include <stdio.h>
#include <time.h>
#include <iostream>
#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>
using namespace std;

#include <opencv2/opencv.hpp>
using namespace cv;

enum ROItype { L_EYE, R_EYE, L_EB, R_EB, MOUTH, NOSE, TEETH, FACE, NONE };

#endif // HEADERS_H
