#include "IntelliCore.h"
#include "App.h"

string IntelliCore::emotionTab[EMOTION_NUM] = {"neutral", "happy", "surprise", "anger", "sad", "disgust", "fear" };

IntelliCore::IntelliCore()
{
    this->neuralNet = new neural_net();
    this->svm = SVM::create();
}

IntelliCore::IntelliCore(string nnPath, string svmPath)
{
    this->neuralNet = new neural_net();
    this->svm = SVM::create();
    if(nnPath != "") loadNN(nnPath);
    if(svmPath != "") loadSVM(svmPath);
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
        string path = App::pathMap["models"] + "nn_" + Logger::getTime();
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
    for (int i = 0; i < testData.length_train_data(); i++)
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

float* IntelliCore::runNN(float* input)
{
    return this->neuralNet->run(input);
}

void IntelliCore::createSVM(int svmType, int kernelType, int gamma)
{
    this->svm = SVM::create();
    this->svm->setType(svmType);
    this->svm->setKernel(kernelType);
    this->svm->setGamma(gamma);
}

void IntelliCore::trainSVM(string dataPath, bool save)
{
    Mat input, target;
    loadDataSVM(dataPath, input, target);
    //Ptr<TrainData> data = TrainData::create(input, ROW_SAMPLE, target);

    cout << endl << "Training SVM..." << endl;
    this->svm->train(input, ROW_SAMPLE, target);
    cout << endl;
    if(save) {;
        string path = App::pathMap["models"] + "svm_" + Logger::getTime();
        cout << "Saving created SVM model: " << path << endl;
        this->svm->save(path);
    }
}

void IntelliCore::testSVM(string testPath)
{
    Mat input, target;
    loadDataSVM(testPath, input, target);

    cout << endl << "Testing SVM..." << endl;
    for (int i = 0; i < input.rows; i++)
    {
        int out = this->svm->predict(input.row(i));

        stringstream ss; ss << i;
        cout << "[" + ss.str() + "]:";
        cout << "\tTarget: " << target.at<int>(i,0);
        cout << endl;
        cout << "\tOut: " << out;
        cout << endl;
    }
    cout << endl;
}

void IntelliCore::loadSVM(string svmPath)
{
    cout << "Loading SVM model from the file: " << svmPath << endl;
    this->svm = Algorithm::load<SVM>(svmPath);
}

void IntelliCore::loadDataSVM(string path, Mat& input, Mat& target)
{
    FILE* file = fopen(path.c_str(), "r");
    if(!file) { cout << "Cannot open the file: " << path << endl; return; }
    float x; int y;
    int sampleNum, inputNum, outputNum;
    fscanf(file, "%d %d %d", &sampleNum, &inputNum, &outputNum);

    input = Mat(sampleNum, inputNum, CV_32F);
    target = Mat(sampleNum, outputNum, CV_32S);

    for(int row = 0; row < sampleNum; row++)
    {
        // Load input values
        for(int col = 0; col < inputNum; col++)
        {
            fscanf(file, "%f ", &x);
            input.at<float>(row, col) = x;
        }

        // Load target values
        for(int col = 0; col < outputNum; col++)
        {
            fscanf(file, "%d", &y);
            target.at<int>(row, col) = y;
        }
    }

    fclose(file);
}

int IntelliCore::runSVM(float* input)
{
    return this->svm->predict(Mat(1, FEAT_NUM, CV_32F, input));
}

IntelliCore::~IntelliCore()
{
    delete this->neuralNet;
}
