#include "IntelliCore.h"
#include "App.h"

IntelliCore::IntelliCore()
{
    this->neuralNet = new neural_net();
}

void IntelliCore::createNN(int inputNum, int hiddenNum, int outputNum)
{       
    this->neuralNet->create_standard(3, inputNum, hiddenNum, outputNum);
    this->neuralNet->set_activation_function_hidden(FANN::SIGMOID);
    this->neuralNet->set_activation_function_output(FANN::SIGMOID);   
}

void IntelliCore::trainNN(string dataPath, int maxEpoch, float desiredError, float learningRate, float momentum, bool save)
{
    training_data trainData = training_data();
    trainData.read_train_from_file(dataPath);
    trainData.shuffle_train_data();
    this->neuralNet->set_learning_rate(learningRate);
    this->neuralNet->set_learning_momentum(momentum);

    cout << endl << "Training network..." << endl;
    this->neuralNet->train_on_data(trainData, maxEpoch, 0.01*maxEpoch, desiredError);
    cout << endl;
    if(save) {;
        string path = App::pathMap["nn_models"] + "nn_" + Logger::getTime();
        cout << "Saving created neural net model: " << path << endl;
        this->neuralNet->save(path);
    }
}

void IntelliCore::testNN(string testPath)
{
    training_data testData = training_data();
    testData.read_train_from_file(testPath);
    testData.shuffle_train_data();

    cout << endl << "Testing network..." << endl;
    cout.setf(ios::fixed); cout << setprecision(3);
    for (int i = 0; i < testData.length_train_data(); ++i)
    {
        fann_type *calc_out = this->neuralNet->run(testData.get_input()[i]);
        stringstream ss; ss << i;
        cout << "[" + ss.str() + "]:";
        cout << "\tTarget: "; for(int j = 0; j < 7; j++) cout << testData.get_output()[i][j] << " ";
        cout << endl;
        cout << "\tOutput: "; for(int j = 0; j < 7; j++) cout << calc_out[j] << " ";
        cout << endl;
    }
    cout << setprecision(6);
    cout.unsetf(ios::fixed | ios::scientific);

    this->neuralNet->test_data(testData);
    cout << "\n\nTest MSE: " << this->neuralNet->get_MSE() << endl;
}

void IntelliCore::loadNN(string nnPath)
{
    cout << "Loading neural net model from the file: " << nnPath << endl;
    this->neuralNet->create_from_file(nnPath);
}

IntelliCore::~IntelliCore()
{
    delete this->neuralNet;
}
