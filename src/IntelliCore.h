#ifndef INTELLICORE_H
#define INTELLICORE_H

#define EMOTION_NUM 7

#include "Headers.h"
#include "Logger.h"

class IntelliCore
{

public:
    IntelliCore();
    IntelliCore(string nnPath);
    ~IntelliCore();
    static string emotionTab[EMOTION_NUM];
    void loadData(string path, Mat& input, Mat& target);

    // run classifier ?

    // Neural net
    void createNN(int inputNum, int hiddenNum, int outputNum);
    void trainNN(string dataPath, int maxEpoch, float desiredError, float learningRate, float momentum, bool save);
    void testNN(string testPath);
    void loadNN(string nnPath);
    double* runNN(double* input);

    // TODO: SVM


private:
    neural_net* neuralNet;

};

#endif // INTELLICORE_H
