// Basing on: http://www.bytefish.de/pdf/machinelearning.pdf
#include "IntelliCore.h"

IntelliCore::IntelliCore()
{
}

void IntelliCore::loadCSV(string path, Mat& input, Mat& target, int N, int X)
{
    FILE* file = fopen(path.c_str(), "r");
    if(!file) { cout << "Cannot open the file: " << path << endl; return; }
    float x;
    int label;

    for(int row = 0; row < N; row++)
    {
        // Load attributes (X) values
        for(int col = 0; col < X; col++)
        {
            fscanf(file, "%f,", &x);
            input.at<float>(row, col) = x;
        }

        // Load class label and set right output node to 1.0
        fscanf(file, "%i", &label);
        target.at<float>(row, label) = 1.0;
    }

    fclose(file);
}

IntelliCore::~IntelliCore()
{
}
