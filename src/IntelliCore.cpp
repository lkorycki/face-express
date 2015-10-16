#include "IntelliCore.h"
#include "App.h"

string IntelliCore::emotionTab[EMOTION_NUM] = { "neutral", "happy", "surprise", "anger", "sad", "disgust", "fear" };

IntelliCore::IntelliCore()
{
    this->neuralNet = new neural_net();
    this->svm = SVM::create();
    this->knn = KNearest::create();
}

IntelliCore::IntelliCore(string nnPath, string svmPath, string knnDataPath)
{
    this->neuralNet = new neural_net();
    this->svm = SVM::create();
    if(nnPath != "") loadNN(nnPath);
    if(svmPath != "") loadSVM(svmPath);
    if(knnDataPath != "") createKNN(knnDataPath, 5);
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
    trainData.shuffle_train_data(); // randomize order

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
    testData.shuffle_train_data(); // randomize order

    Mat response = Mat::zeros(EMOTION_NUM, EMOTION_NUM+1, CV_32S);
    int pos = 0;

    cout << "Testing neural network..." << endl;
    int sampleNum = testData.length_train_data();
    for (int i = 0; i < sampleNum; i++)
    {
        fann_type *calc_out = this->neuralNet->run(testData.get_input()[i]);

        int tar = MathCore::maxPos(testData.get_output()[i], EMOTION_NUM);
        int out = MathCore::maxPos(calc_out, EMOTION_NUM);

        response.at<int>(tar, out)++; // target in row, actual response in column
        response.at<int>(tar, EMOTION_NUM)++; // sum of target samples
        if(out == tar) pos++;
    }

    cout.setf(ios::fixed); cout << setprecision(3);

    cout << "Responses:\n" << response << "\n\n";
    cout << "Accuracy: \n";
    for(int i = 0; i < EMOTION_NUM; i++)
        cout << "\t" << emotionTab[i] << " : " << 100*(float)response.at<int>(i,i) / response.at<int>(i,EMOTION_NUM) << "%\n";
    cout << endl << "\tGENERAL: " << 100*(float)pos / sampleNum << "%\n";

    this->neuralNet->test_data(testData);
    cout << "\tMSE: " << this->neuralNet->get_MSE() << endl;

    cout << setprecision(6);
    cout.unsetf(ios::fixed | ios::scientific);
}

void IntelliCore::loadNN(string nnPath)
{
    cout << "Loading neural net model from the file: " << nnPath << endl;
    this->neuralNet->create_from_file(nnPath);
}

inline float* IntelliCore::runNN(float* input)
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

void IntelliCore::trainModel(StatModel* model, string dataPath, bool save)
{
    Mat input, target;
    loadDataCV(dataPath, input, target);

    cout << "Training model..." << endl;
    model->train(input, ROW_SAMPLE, target);
    cout << endl;
    if(save) {;
        string path = App::pathMap["models"] + "svm_" + Logger::getTime();
        cout << "Saving created model: " << path << endl;
        model->save(path);
    }
}

void IntelliCore::testModel(StatModel* model, string testPath, bool ensemble)
{
    Mat input, target, response = Mat::zeros(EMOTION_NUM, EMOTION_NUM+1, CV_32S);
    loadDataCV(testPath, input, target);
    int pos = 0;

    cout << "Testing model..." << endl;
    for (int i = 0; i < input.rows; i++)
    {
        int out;
        if(!ensemble) out = model->predict(input.row(i));
        else out = (int)runEnsemble(input.ptr<float>(i))[0];
        int tar = target.at<int>(i,0);

        response.at<int>(tar-1, out-1)++; // target in row, actual response in column
        response.at<int>(tar-1, EMOTION_NUM)++; // sum of target samples
        if(out == tar) pos++;
    }

    cout.setf(ios::fixed); cout << setprecision(3);

    cout << "Responses:\n" << response << "\n\n";
    cout << "Accuracy: \n";
    for(int i = 0; i < EMOTION_NUM; i++)
        cout << "\t" << emotionTab[i] << " : " << 100*(float)response.at<int>(i,i) / response.at<int>(i,EMOTION_NUM) << "%\n";
    cout << endl << "\tGENERAL: " << 100*(float)pos / input.rows << "%\n";

    cout << setprecision(6);
    cout.unsetf(ios::fixed | ios::scientific);
}

void IntelliCore::loadSVM(string modelPath)
{
    cout << "Loading SVM model from the file: " << modelPath << endl;
    this->svm = Algorithm::load<ml::SVM>(modelPath);
}

void IntelliCore::loadDataCV(string path, Mat& input, Mat& target)
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
            fscanf(file, "%f", &x);
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

inline float* IntelliCore::runModel(StatModel* model, float* input)
{
    return new float[1] { model->predict(Mat(1, FEAT_NUM, CV_32F, input)) };
}

void IntelliCore::createKNN(string dataPath, int k)
{
    this->knn = KNearest::create();
    this->knn->setDefaultK(k);
    this->knn->setIsClassifier(true);

    cout << "Creating k-NN with data: " << dataPath << endl;
    trainModel(this->knn, dataPath, false);
}

float* IntelliCore::runEnsemble(float* input)
{
    // Simple classifiers ensembling - majority voting
    int votes[7] = { 0 };

    float* nnResult = runNN(input);
    int max = MathCore::maxPos(nnResult, EMOTION_NUM);

    votes[max]++;
    votes[(int)runModel(this->svm, input)[0]-1]++;
    votes[(int)runModel(this->knn, input)[0]-1]++;

    return new float[1] { MathCore::maxPos(votes, EMOTION_NUM)+1 };
}

float* IntelliCore::runClassifier(ClassifierType cType, float* input)
{
    switch(cType)
    {
        case ClassifierType::NN:
            return runNN(input); break;
        case ClassifierType::SVM:
            return runModel(this->svm, input); break;
        case ClassifierType::KNN:
            return runModel(this->knn, input); break;
        case ClassifierType::ENSEMBLE:
            return runEnsemble(input); break;
        default:
            return new float[1] { -1 };
    }
}

IntelliCore::~IntelliCore()
{
    delete this->neuralNet;
}
