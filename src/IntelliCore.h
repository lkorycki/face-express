#ifndef INTELLICORE_H
#define INTELLICORE_H

#define EMOTION_NUM 7

#include "Headers.h"
#include "Logger.h"

class IntelliCore
{

public:
    IntelliCore();
    IntelliCore(string nnPath, string svmPath, string knnDataPath);
    ~IntelliCore();
    static string emotionTab[EMOTION_NUM];
    void loadDataCV(string path, Mat& input, Mat& target);
    float* runClassifier(ClassifierType cType, float* input);

    // Neural net
    void createNN(int inputNum, int hiddenNum, int outputNum);
    void trainNN(string dataPath, int maxEpoch, float desiredError, float learningRate, float momentum, bool save);
    void testNN(string testPath);
    void loadNN(string nnPath);
    inline float* runNN(float* input);

    // SVM and StatModels (for OpenCV models, e.g. k-NN + ensemble)
    void createSVM(int svmType, int kernelType, int gamma = 0);
    void loadSVM(string modelPath);
    void trainModel(StatModel* model, string dataPath, bool save);
    void testModel(StatModel* model, string testPath, bool ensemble = false);
    inline float* runModel(StatModel* model, float* input);

    // k-NN
    void createKNN(string dataPath, int k);

    // Ensemble
    float* runEnsemble(float* input);

//private:
    neural_net* neuralNet;
    Ptr<ml::SVM> svm;
    Ptr<ml::KNearest> knn;

};

#endif // INTELLICORE_H
