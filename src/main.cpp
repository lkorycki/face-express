#include "Headers.h"
#include "App.h"

int main(int argc, char *argv[])
{
    App app = App();
    app.runCam(0);

    //IntelliCore ic = IntelliCore();
    //ic->loadNN("/home/lukas/Projects/PD/stat_models/nn_model_double");
    //ic.createNN(16,7,7);
    //ic.trainNN("/home/lukas/Projects/PD/training_data/training_vectors/vec_fann.data", 1000, 0.01, 0.2, 0.05, true);
    //ic.testNN("/home/lukas/Projects/PD/training_data/test_vectors/vec_fann.data");

    return 0;
}


