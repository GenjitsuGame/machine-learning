#include <iostream>
#include <cassert>


int sign(const float &input) {
    if (input >= 0) return 1;
    else return -1;
}

int get_random_example(float *inputs) {
    assert(inputs && *inputs);

    srand((unsigned int) time(NULL));
    const int size = (sizeof(inputs) / sizeof(*inputs));
    return rand() % size;
}

template<typename T>
T &two_dim_get(const T *array, const int &width, int &x, int &y) {
    return array[width * y + x];
}

template<typename T>
void two_dim_set(const T *&array, const int &width, int &x, int &y, T e) {
    array[width * y + x] = e;
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