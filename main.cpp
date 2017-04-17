#include <iostream>
#include <cassert>
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
    return (double) (1.f / (1.f + exp(-x)));
}

double activation_derivative(double &x) {
    return activation(x) * (1 - activation(x));
}

double get_random_double(double min, double max) {
    return min + static_cast <double> (rand()) / ((RAND_MAX / (max - min)));
}

double **get_random_model(int *inputs, int inputsSize) {
    assert(inputs && *inputs && inputsSize > 0);

    double **model = new double *[inputsSize];

    for (int i = 0; i < inputsSize; ++i) {
        if (inputs[i] <= 0) {
            throw new std::out_of_range("Number of inputs must be greater than 0.");
        }

        model[i] = new double[inputs[i]];
        for (int j = 0; j < inputs[i]; ++j) {
            model[i][j] = get_random_double(-1.f, 1.f);
        }
    }

    return model;
}

template<typename T>
T *two_dim_get(T *&array, int &width, int &x, int &y) {
    return (array + width * y + x);
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

    return rand() % (examplesSize / inputSize);
}


template<typename T>
T **get_example_at(T *examples, int inputSize, int pos) {
    T **example = new T *[inputSize];

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

double linear_regression(double *model, double **inputs, int inputSize) {
    assert(model);
    assert(inputs);

    double weightedSum = 0;

    for (int i = 0; i < inputSize; ++i) {
        weightedSum += *(inputs[i]) * model[i];
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
linear_classification(double *model, double **inputs, int inputSize) {
    assert(model && *model);
    assert(inputs && *inputs);

    return sign(linear_regression(model, inputs, inputSize));
}

double *
create_linear_classification_model(double *examples, int examplesSize, int inputSize,
                                   int *desiredOutputs, int epochs, double learningRate) {
    assert(examples);
    assert(desiredOutputs);
    assert(inputSize > 0);
    assert(epochs > 0);

    double *model = get_random_model(new int[1]{inputSize}, 1)[0];

    for (int i = 0; i < epochs; ++i) {
        int pos = get_random_example_pos(examples, examplesSize, inputSize);
        double **example = get_example_at(examples, inputSize, pos);
        int output = linear_classification(model, example, inputSize);
        if (output != desiredOutputs[pos]) {
            perceptron_learning_algorithm(model, *example, inputSize, desiredOutputs[pos], learningRate);
        }
    }

    return model;
}

int mlp_linear_regression();

// TEST

int regressionExamplesSize = 4;
int regressionExampleSize = 1;
double *regressionExamples = new double[regressionExamplesSize * regressionExampleSize]{100, 200, 400, 500};
double *regressionDesiredOutputs = new double[regressionExamplesSize * regressionExampleSize]{200, 400, 800,
                                                                                              1000};

int classificationExamplesSize = 4;
int classificationExampleSize = 2;
double *classificationExamples = new double[classificationExamplesSize * classificationExampleSize]{-1, -1,
                                                                                                    -1,
                                                                                                    1, 1,
                                                                                                    -1,
                                                                                                    1, 1};
int *classificationDesiredOutputs = new int[classificationExamplesSize]{-1, -1, 1, 1};

void test_get_example() {
    double **example = get_example_at(classificationExamples, classificationExampleSize, 1);
    assert(*example == (classificationExamples + 2));
}

void test_get_random_model() {
    int layersSize = 3;
    int *layersSizes = new int[layersSize]{4, 3, 5};

    double **model = get_random_model(layersSizes, layersSize);
    for (int i = 0; i < layersSize; ++i) {
        for (int j = 0; j < layersSizes[i]; ++j) {
            assert(model[i][j]);
        }
        delete[] model[i];
    }

    delete[] model;
    delete[] layersSizes;
}

void test_linear_regression() {
    double *model = new double[regressionExampleSize]{2};
    double input = 300;
    double **inputs = new double *[1]{&input};
    double output = linear_regression(model, inputs, regressionExampleSize);

    assert(output == 600);
}

void test_create_linear_regression_model() {
    double *model = create_linear_regression_model(regressionExamples, regressionExamplesSize, regressionExampleSize,
                                                   regressionDesiredOutputs);
    assert(1.99999 <= model[0] <= 2);
}

void test_linear_classification() {
    int layersSize = 1;
    int *layersSizes = new int[layersSize]{1};
    int modelSize = 2;
    double *model = new double[modelSize]{1, -1};
    int desiredOutput = -1;
    int examplePos = 1;
    double **example = get_example_at(classificationExamples, classificationExampleSize, examplePos);
    int output = linear_classification(model, example, classificationExampleSize);

    assert(desiredOutput == output);

    delete[] layersSizes;
    delete[] model;
}

void test_create_linear_classification_model() {
    int epochs = 2000;
    int desiredOutput = -1;
    int examplePos = 1;
    double learningRate = 0.01;
    double **example = get_example_at(classificationExamples, classificationExampleSize, examplePos);
    int output = linear_classification(
            create_linear_classification_model(classificationExamples, classificationExamplesSize,
                                               classificationExampleSize, classificationDesiredOutputs, epochs,
                                               learningRate), example, classificationExampleSize);

    assert(desiredOutput == output);
}

void test() {
    init();
    test_get_example();
    test_get_random_model();
    test_linear_regression();
    test_create_linear_regression_model();
    test_linear_classification();
    test_create_linear_classification_model();
}

int main() {
    test();
    return 0;
}