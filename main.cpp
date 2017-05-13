#include <iostream>
#include <cmath>
#include "Eigen/Core"
#include "Eigen/Dense"

void init() {
    srand(static_cast <unsigned> (time(0)));
    std::cout << std::boolalpha;
    std::cout.precision(25);
}

int sign(double input) {
    if (input >= 0) return 1;
    else return -1;
}

double activation(double &x) {
    return tanh(x);
}

double activation_derivative(double &x) {
    return activation(x) * (1 - activation(x));
}

double get_random_double(double min, double max) {
    return min + static_cast <double> (rand()) / ((RAND_MAX / (max - min)));
}

void print_outputs(double **outputs, int *modelStruct, int modelStructSize) {
    for (int i = 0; i < modelStructSize; ++i) {
        for (int j = 0; j < modelStruct[i]; ++j) {
            std::cout << outputs[i][j] << '\t';
        }
        std::cout << std::endl;
    }
}

void print_model(double ***model, int *modelStruct, int modelStructSize, int inputsSize) {
    for (int i = 0; i < modelStructSize; ++i) {
        for (int j = 0; j < modelStruct[i]; ++j) {
            for (int k = 0; k < inputsSize; ++k) {
                std::cout << model[i][j][k] << '\t';
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

double ***get_random_model(int *modelStruct, int modelStructSize, int inputsSize) {
    assert(modelStruct != nullptr);
    assert(modelStructSize > 0);
    assert(inputsSize > 0);

    double ***model = new double **[modelStructSize];

    for (int i = 0; i < modelStructSize; ++i) {
        if (modelStruct[i] <= 0) {
            throw new std::out_of_range("Number of neurons must be greater than 0.");
        }
        model[i] = new double *[modelStruct[i]];
        for (int j = 0; j < modelStruct[i]; ++j) {
            int prevInputsSize = i == 0 ? inputsSize : modelStruct[i - 1];
            model[i][j] = new double[prevInputsSize];
            for (int k = 0; k < prevInputsSize; ++k) {
                model[i][j][k] = get_random_double(-1.f, 1.f);
            }
        }
    }
    return model;
}

template<typename T>
T two_dim_get(T *&array, int &width, int &x, int &y) {
    return array[width * y + x];
}

template<typename T>
void two_dim_set(T *&array, int &width, int x, int y, T e) {
    array[width * y + x] = e;
}

template<typename T>
int get_random_example_pos(T *examples, int examplesSize, int inputSize) {
    assert(examples);
    assert(examplesSize > 0);
    assert(inputSize > 0);

    return rand() % examplesSize;
}


template<typename T>
T *get_example_at(T *examples, int inputSize, int pos) {
    T *example = new T[inputSize];

    for (int i = 0; i < inputSize; ++i) {
        example[i] = two_dim_get(examples, inputSize, i, pos);
    }

    return example;
}

void
perceptron_learning_algorithm(double *model, double *inputs, int inputsSize,
                              int desiredOutput, double learningRate) {
    assert(inputs);
    assert(inputsSize > 0);
    assert(model);
    assert(learningRate > 0);

    for (int i = 0; i < inputsSize; ++i) {
        model[i] += learningRate * inputs[i] * desiredOutput;
    }
}

double linear_regression(double *model, double *inputs, int inputsSize) {
    assert(model);
    assert(inputs);

    double weightedSum = 0;

    for (int i = 0; i < inputsSize; ++i) {
        weightedSum += inputs[i] * model[i];
    }

    return weightedSum;
}

double *
create_linear_regression_model(double *examples, int examplesSize, int inputSize,
                               double *desiredOutputs) {
    assert(examples);
    assert(desiredOutputs);
    assert(inputSize > 0);

    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mExamples(examples, examplesSize,
                                                                                                 inputSize);
    Eigen::MatrixXd tmExamples = mExamples.transpose();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mOutputs(desiredOutputs,
                                                                                                examplesSize, 1);

    Eigen::MatrixXd product1(tmExamples * mExamples);

    Eigen::MatrixXd result = product1.inverse() * tmExamples * mOutputs;

    double *resultArray = new double[result.size()];
    for (int i = 0; i < result.size(); ++i) {
        resultArray[i] = result.data()[i];
    }

    return resultArray;
}

int
linear_classification(double *model, double *inputs, int inputSize) {
    assert(model && *model);
    assert(inputs && *inputs);

    return sign(linear_regression(model, inputs, inputSize));
}

double *
create_linear_classification_model(double *examples, int examplesSize, int inputsSize,
                                   int *desiredOutputs, int epochs, double learningRate) {
    assert(examples);
    assert(desiredOutputs);
    assert(inputsSize > 0);
    assert(epochs > 0);

    double *model = get_random_model(new int[1]{inputsSize}, 1, inputsSize)[0][0];

    for (int i = 0; i < epochs; ++i) {
        int pos = get_random_example_pos(examples, examplesSize, inputsSize);
        double *example = get_example_at(examples, inputsSize, pos);
        int output = linear_classification(model, example, inputsSize);
        if (output != desiredOutputs[pos]) {
            perceptron_learning_algorithm(model, example, inputsSize, desiredOutputs[pos], learningRate);
        }
    }

    return model;
}

double *add_bias_input(double *inputs, int inputsSize) {
    double *inputsWithBias = new double[inputsSize + 1];
    for (int i = 0; i < inputsSize; ++i) {
        inputsWithBias[i] = inputs[i];
    }
    inputsWithBias[inputsSize] = 1;
    return inputsWithBias;
}

double **mlp_regression_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                     int inputsSize) {
    double **outputs = new double *[modelStructSize];
    double *inputsWithBias = add_bias_input(inputs, inputsSize);
    inputsSize += 1;
    for (int i = 0; i < modelStructSize - 1; ++i) {
        outputs[i] = new double[modelStruct[i]]{0};
        for (int j = 0; j < modelStruct[i] - 1; ++j) {
            if (i == 0) {
                for (int k = 0; k < inputsSize; ++k) {
                    outputs[i][j] += inputsWithBias[k] * model[i][j][k];
                }
            } else {
                for (int k = 0; k < modelStruct[i - 1]; ++k) {
                    outputs[i][j] += outputs[i - 1][k] * model[i][j][k];
                }
            }
        }
        outputs[i][modelStruct[i] - 1] = 1;
    }

    outputs[modelStructSize - 1] = new double[modelStruct[modelStructSize - 1]];
    outputs[modelStructSize - 1][0] = 0;
    for (int i = 0; i < modelStruct[modelStructSize - 2]; ++i) {
        outputs[modelStructSize - 1][0] += outputs[modelStructSize - 2][i] * model[modelStructSize - 1][0][i];
    }

    return outputs;
}


double **mlp_classification_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                         int inputsSize) {
    double **outputs = new double *[modelStructSize];
    double *inputsWithBias = add_bias_input(inputs, inputsSize);
    inputsSize += 1;
    for (int i = 0; i < modelStructSize - 1; ++i) {
        outputs[i] = new double[modelStruct[i]]{0};
        for (int j = 0; j < modelStruct[i] - 1; ++j) {
            if (i == 0) {
                for (int k = 0; k < inputsSize; ++k) {
                    outputs[i][j] += inputsWithBias[k] * model[i][j][k];
                }
            } else {
                for (int k = 0; k < modelStruct[i - 1]; ++k) {
                    outputs[i][j] += outputs[i - 1][k] * model[i][j][k];
                }
            }
            outputs[i][j] = activation(outputs[i][j]);
        }
        outputs[i][modelStruct[i] - 1] = 1;
    }

    outputs[modelStructSize - 1] = new double[modelStruct[modelStructSize - 1]];
    outputs[modelStructSize - 1][0] = 0;
    for (int i = 0; i < modelStruct[modelStructSize - 2]; ++i) {
        outputs[modelStructSize - 1][0] += outputs[modelStructSize - 2][i] * model[modelStructSize - 1][0][i];
    }

    outputs[modelStructSize - 1][0] = activation(outputs[modelStructSize - 1][0]);
    return outputs;
}


void mlp_update_weight(double &weight, double &learningRate, double &output, double &error) {
    weight = weight - learningRate * output * error;
}


void mlp_regression_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                   int inputsSize, double **outputs, double *desiredOutputs, double &learningRate) {
    double **errors = new double *[modelStructSize];

    int lastLayer = modelStructSize - 1;
    errors[lastLayer] = new double[modelStruct[lastLayer]];
    for (int i = 0; i < modelStruct[lastLayer]; ++i) {
        errors[lastLayer][i] = outputs[lastLayer][i] - desiredOutputs[i];
    }

    for (int i = lastLayer - 1; i >= 0; --i) {
        errors[i] = new double[modelStruct[i]];
        for (int j = 0; j < modelStruct[i]; ++j) {
            int error = 0;
            for (int k = 0; k < modelStruct[i + 1]; ++k) {
                error += model[i + 1][k][j] * errors[i + 1][k];
            }
            errors[i][j] = (1 - pow(outputs[i][j], 2)) * error;
        }
    }

    for (int i = 0; i < modelStructSize; ++i) {
        for (int j = 0; j < modelStruct[i]; ++j) {
            if (i == 0) {
                for (int k = 0; k < inputsSize; ++k) {
                    mlp_update_weight(model[i][j][k], learningRate, inputs[k], errors[i][j]);
                }
            } else {
                for (int k = 0; k < modelStruct[i - 1]; ++k) {
                    mlp_update_weight(model[i][j][k], learningRate, outputs[i - 1][k], errors[i][j]);
                }
            }
        }
    }
}

void mlp_classification_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                       int inputsSize, double **outputs, int *desiredOutputs, double &learningRate) {
    double **errors = new double *[modelStructSize];

    int lastLayer = modelStructSize - 1;
    errors[lastLayer] = new double[modelStruct[lastLayer]];
    for (int i = 0; i < modelStruct[lastLayer]; ++i) {
        errors[lastLayer][i] = (1 - pow(outputs[lastLayer][i], 2)) * (outputs[lastLayer][i] - desiredOutputs[i]);
    }

    for (int i = lastLayer - 1; i >= 0; --i) {
        errors[i] = new double[modelStruct[i]];
        for (int j = 0; j < modelStruct[i]; ++j) {
            int error = 0;
            for (int k = 0; k < modelStruct[i + 1]; ++k) {
                error += model[i + 1][k][j] * errors[i + 1][k];
            }
            errors[i][j] = (1 - pow(outputs[i][j], 2)) * error;
        }
    }

    for (int i = 0; i < modelStructSize; ++i) {
        for (int j = 0; j < modelStruct[i]; ++j) {
            if (i == 0) {
                for (int k = 0; k < inputsSize; ++k) {
                    mlp_update_weight(model[i][j][k], learningRate, inputs[k], errors[i][j]);
                }
            } else {
                for (int k = 0; k < modelStruct[i - 1]; ++k) {
                    mlp_update_weight(model[i][j][k], learningRate, outputs[i - 1][k], errors[i][j]);
                }
            }
        }
    }
}

void mlp_classification_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
                            int *desiredOutput, double learningRate) {
    double **outputs = mlp_classification_feed_forward(model, modelStruct, modelStructSize, inputs, inputsSize);
    mlp_classification_back_propagate(model, modelStruct, modelStructSize, inputs, inputsSize, outputs, desiredOutput,
                                      learningRate);
}

void mlp_regression_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
                        double *desiredOutput, double learningRate) {
    double **outputs = mlp_regression_feed_forward(model, modelStruct, modelStructSize, inputs, inputsSize);
    mlp_regression_back_propagate(model, modelStruct, modelStructSize, inputs, inputsSize, outputs, desiredOutput,
                                  learningRate);
}

void
mlp_classification_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                         int inputsSize,
                         int **desiredOutputs, double learningRate, int epochs) {
    for (int i = 0; i < epochs; ++i) {
        int pos = get_random_example_pos(examples, examplesSize, inputsSize);
        double *example = get_example_at(examples, inputsSize, pos);
        mlp_classification_fit(model, modelStruct, modelStructSize, example, inputsSize, desiredOutputs[pos],
                               learningRate);
    }
}

void
mlp_regression_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                     int inputsSize,
                     double **desiredOutputs, double learningRate, int epochs) {
    for (int i = 0; i < epochs; ++i) {
        int pos = get_random_example_pos(examples, examplesSize, inputsSize);
        double *example = get_example_at(examples, inputsSize, pos);
        mlp_regression_fit(model, modelStruct, modelStructSize, example, inputsSize, desiredOutputs[pos],
                           learningRate);
    }
}

double ***mlp_classification_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                                          int inputsSize,
                                          int **desiredOutputs, double learningRate, int epochs) {
    double ***model = get_random_model(modelStruct, modelStructSize, inputsSize);
    mlp_classification_train(model, modelStruct, modelStructSize, examples, examplesSize, inputsSize, desiredOutputs,
                             learningRate, epochs);

    return model;
}

double ***mlp_regression_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                                      int inputsSize,
                                      double **desiredOutputs, double learningRate, int epochs) {
    double ***model = get_random_model(modelStruct, modelStructSize, inputsSize);
    mlp_regression_train(model, modelStruct, modelStructSize, examples, examplesSize, inputsSize, desiredOutputs,
                         learningRate, epochs);

    return model;
}


// TEST

void test_get_example() {
    int classificationExamplesSize = 4;
    int classificationExampleSize = 2;
    double *classificationExamples = new double[classificationExamplesSize * classificationExampleSize]{-1, -1,
                                                                                                        -1,
                                                                                                        1, 1,
                                                                                                        -1,
                                                                                                        1, 1};

    double *example = get_example_at(classificationExamples, classificationExampleSize, 1);
    assert(example[0] == classificationExamples[2]);

    delete[] classificationExamples;
}

void test_get_random_model() {
    int inputsSize = 2;
    int modelStructSize = 3;
    int *modelStruct = new int[modelStructSize]{4, 3, 5};

    double ***model = get_random_model(modelStruct, modelStructSize, inputsSize);
    for (int i = 0; i < modelStructSize; ++i) {
        for (int j = 0; j < modelStruct[i]; ++j) {
            for (int k = 0; k < inputsSize; ++k) {
                assert(-1 <= model[i][j][k] && model[i][j][k] <= 1);
            }
            delete[] model[i][j];
        }
        delete[] model[i];
    }

    delete[] model;
    delete[] modelStruct;
}

void test_linear_regression() {
    int regressionExampleSize = 1;

    double *model = new double[regressionExampleSize]{2};
    double input = 300;
    double *inputs = new double[1]{input};
    double output = linear_regression(model, inputs, regressionExampleSize);

    assert(output == 600);
}

void test_create_linear_regression_model() {
    int regressionExamplesSize = 4;
    int regressionExampleSize = 1;
    double *regressionExamples = new double[regressionExamplesSize * regressionExampleSize]{100, 200, 400, 500};
    double *regressionDesiredOutputs = new double[regressionExamplesSize * regressionExampleSize]{200, 400, 800,
                                                                                                  1000};

    double *model = create_linear_regression_model(regressionExamples, regressionExamplesSize, regressionExampleSize,
                                                   regressionDesiredOutputs);
    assert(1.99999 <= model[0] && model[0] <= 2);
    delete[] regressionExamples;
    delete[] regressionDesiredOutputs;
}

void test_linear_classification() {
    int classificationExamplesSize = 4;
    int classificationExampleSize = 2;
    double *classificationExamples = new double[classificationExamplesSize * classificationExampleSize]{-1, -1,
                                                                                                        -1,
                                                                                                        1, 1,
                                                                                                        -1,
                                                                                                        1, 1};

    int layersSize = 1;
    int *layersSizes = new int[layersSize]{1};
    int modelSize = 2;
    double *model = new double[modelSize]{1, -1};
    int desiredOutput = -1;
    int examplePos = 1;
    double *example = get_example_at(classificationExamples, classificationExampleSize, examplePos);
    int output = linear_classification(model, example, classificationExampleSize);

    assert(desiredOutput == output);

    delete[] layersSizes;
    delete[] model;
    delete[] classificationExamples;
}

void test_create_linear_classification_model() {

    int classificationExamplesSize = 4;
    int classificationExampleSize = 2;
    double *classificationExamples = new double[classificationExamplesSize * classificationExampleSize]{-1, -1,
                                                                                                        -1,
                                                                                                        1, 1,
                                                                                                        -1,
                                                                                                        1, 1};
    int *classificationDesiredOutputs = new int[classificationExamplesSize]{-1, -1, 1, 1};

    int epochs = 100000;
    int desiredOutput = -1;
    int examplePos = 1;
    double learningRate = 0.01;
    double *example = get_example_at(classificationExamples, classificationExampleSize, examplePos);
    int output = linear_classification(
            create_linear_classification_model(classificationExamples, classificationExamplesSize,
                                               classificationExampleSize, classificationDesiredOutputs, epochs,
                                               learningRate), example, classificationExampleSize);

    assert(desiredOutput == output);
    delete[] classificationExamples;
}

void test_mlp_classification() {
    int classificationExamplesSize = 4;
    int classificationExampleSize = 2;
    double *classificationExamples = new double[classificationExamplesSize * classificationExampleSize]{-0.2, -0.2,
                                                                                                        -0.2,
                                                                                                        0.2, 0.2,
                                                                                                        0.2,
                                                                                                        0.2, -0.2};
    int **classificationDesiredOutputs = new int *[classificationExamplesSize]{
            new int[1]{-1},
            new int[1]{-1},
            new int[1]{1},
            new int[1]{1}
    };
    int modelStructSize = 3;
    int *modelStruct = new int[modelStructSize]{3, 4, 1};


    double ***model = mlp_classification_create_model(modelStruct, modelStructSize, classificationExamples,
                                                      classificationExamplesSize,
                                                      classificationExampleSize,
                                                      classificationDesiredOutputs, 0.01, 10000);
    double **outputs = mlp_classification_feed_forward(model, modelStruct, modelStructSize, new double[2]{1, 1}, 2);


    std::cout << outputs[2][0] << std::endl;
}

void test_mlp_regression() {
    int regressionExamplesSize = 4;
    int regressionExampleSize = 1;
    double *regressionExamples = new double[regressionExamplesSize * regressionExampleSize]{100, 200, 400, 500};
    double **regressionDesiredOutputs = new double *[regressionExamplesSize * regressionExampleSize]{
            new double[1]{200},
            new double[1]{400},
            new double[1]{800},
            new double[1]{1000}
    };
    int modelStructSize = 3;
    int *modelStruct = new int[modelStructSize]{3, 4, 1};

    double ***model = mlp_regression_create_model(modelStruct, modelStructSize, regressionExamples,
                                                  regressionExamplesSize,
                                                  regressionExampleSize,
                                                  regressionDesiredOutputs, 0.01, 10000);

    double **outputs = mlp_regression_feed_forward(model, modelStruct, modelStructSize, new double[1]{300}, 1);
    std::cout << outputs[2][0] << std::endl;
}


void test() {
    init();
    test_get_example();
    test_get_random_model();
    test_linear_regression();
    test_create_linear_regression_model();
    test_linear_classification();
    test_create_linear_classification_model();
    test_mlp_classification();
    test_mlp_regression();
}

int main() {
    test();
    return 0;
}