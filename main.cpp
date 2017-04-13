#include <iostream>
#include <cassert>

void init() {
    srand(static_cast <unsigned> (time(NULL)));
}

int sign(const float &input) {
    if (input >= 0) return 1;
    else return -1;
}

int get_random_example(const float * const inputs, const int inputSize) {
    assert(inputs && *inputs);

    int inputsSize = (sizeof(inputs) / sizeof(*inputs));
    assert(inputsSize % inputSize == 0);


    return rand() % (inputsSize / inputSize);
}

float get_random_float(float min, float max) {
    return min + static_cast <float> (rand()) / ((RAND_MAX / (max - min)));
}

float **get_random_model(const int * const inputs) {
    assert(inputs && *inputs);

    const int inputsSize = (sizeof(inputs) / sizeof(*inputs));
    float **const model = new float *[inputsSize];

    for (int i = 0; i < inputsSize; ++i) {
        if (inputs[i] <= 0) {
            throw new std::out_of_range("Number of inputs must be greater than 0.");
        }

        model[i] = new float[inputs[i]];
        for (int j = 0; j < inputs[i]; ++j) {
            model[i][j] = get_random_float(-1.f, 1.f);
        }
    }

    delete [] inputs;
    return model;
}

template<typename T>
T &two_dim_get(const T *array, const int &width, int &x, int &y) {
    return array[width * y + x];
}

template<typename T>
void two_dim_set(const T *&array, const int &width, int &x, int &y, T e) {
    array[width * y + x] = e;
}

float *create_linear_regression_model(float * const examples, const int inputSize, float * const desiredOutputs) {
    assert(examples && *examples);
    assert(desiredOutputs && *desiredOutputs);
    assert(inputSize > 0);

    float ** randomModel = get_random_model(new int[1]{inputSize});

}


float linear_regression(float *model, float *input) {
    assert(model && *model);
    assert(input && *input);

    const int modelSize = (sizeof(model) / sizeof(*model));
    const int inputSize = (sizeof(input) / sizeof(*input));
    assert(modelSize == inputSize);

    float weightedSum = 0;

    for (int i = 0; i < inputSize; ++i) {
        weightedSum += input[i] * model[i];
    }

    return weightedSum;
}

float linear_classification(float *model, float *input) {
    assert(model && *model);
    assert(input && *input);

    return sign(linear_regression(model, input));
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}