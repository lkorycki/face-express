#ifndef INTELLICORE_H
#define INTELLICORE_H

#define EMOTION_NUM 7

#include "Headers.h"
#include "Logger.h"

class IntelliCore
{

public:
    IntelliCore();
    IntelliCore(string nnPath, string svmPath);
    ~IntelliCore();
    static string emotionTab[EMOTION_NUM];

    // run classifier ?

    // Neural net
    void createNN(int inputNum, int hiddenNum, int outputNum);
    void trainNN(string dataPath, int maxEpoch, float desiredError, float learningRate, float momentum, bool save);
    void testNN(string testPath);
    void loadNN(string nnPath);
    float* runNN(float* input);

    // SVM
    void createSVM(int svmType, int kernelType, int gamma = 0);
    void trainSVM(string dataPath, bool save);
    void testSVM(string testPath);
    void loadSVM(string svmPath);
    void loadDataSVM(string path, Mat& input, Mat& target);
    int runSVM(float* input);

private:
    neural_net* neuralNet;
    Ptr<SVM> svm;

};

#endif // INTELLICORE_H
