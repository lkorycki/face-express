#include "Headers.h"
#include "App.h"

int main(int argc, char *argv[])
{
    App app = App();
    app.runCam(0);

//    IntelliCore ic = IntelliCore("/home/lukas/Projects/PD/models/nn_model_float",
//                                  "/home/lukas/Projects/PD/models/svm_model",
//                                  "/home/lukas/Projects/PD/training_data/training_vectors/vec_svm.data");
//            ic.testNN("/home/lukas/Projects/PD/training_data/test_vectors/vec_fann.data");
//            ic.testModel(ic.svm, "/home/lukas/Projects/PD/training_data/test_vectors/vec_svm.data");
//            ic.testModel(ic.knn, "/home/lukas/Projects/PD/training_data/test_vectors/vec_svm.data");
//            ic.testModel(NULL, "/home/lukas/Projects/PD/training_data/test_vectors/vec_svm.data", true);

    //ic.loadSVM("/home/lukas/Projects/PD/models/svm_model");
    //ic.createSVM(SVM::C_SVC, SVM::RBF, 10);
    //ic.trainSVM("/home/lukas/Projects/PD/training_data/training_vectors/vec_svm.data", true);
    //ic.testModel(ic.svm, "/home/lukas/Projects/PD/training_data/test_vectors/vec_svm.data");
    //float in[] = {0.228814,0.169492,0.389831,0.177966,0.372881,0.322034,0.305085,0.364407,0.220339,0.313559,0.79661,0.313559,0.677966,0.525424,0.525424,0.0043315};
    //cout << "Run: " << IntelliCore::emotionTab[ic.runSVM(in)-1];

    //ic.loadNN("/home/lukas/Projects/PD/models/nn_model_float");
    //ic.createNN(16,7,7);
    //ic.trainNN("/home/lukas/Projects/PD/training_data/training_vectors/vec_fann.data", 1000, 0.01, 0.2, 0.05, true);
    //ic.testNN("/home/lukas/Projects/PD/training_data/test_vectors/vec_fann.data");

    return 0;
}


