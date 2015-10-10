#ifndef INTELLICORE_H
#define INTELLICORE_H

#include "Headers.h"
#include "Logger.h"
class App;

class IntelliCore
{

public:
    IntelliCore();
    ~IntelliCore();

    void createNN(int inputNum, int hiddenNum, int outputNum);
    void trainNN(string dataPath, int maxEpoch, float desiredError, float learningRate, float momentum, bool save);
    void testNN(string testPath);
    void loadNN(string nnPath);

private:
    neural_net* neuralNet;

};

#endif // INTELLICORE_H
