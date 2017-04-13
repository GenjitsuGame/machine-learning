#include <iostream>
#include <cassert>


int sign(float input) {
    if (input >= 0) return 1;
    else return -1;
}

void linear_regression(float *model, float *input) {
    assert(model && *model);
    assert(input && *input);

    const int modelSize = (sizeof(model) / sizeof(*model));
    const int inputSize = (sizeof(input) / sizeof(*input));
    int *outputs = new int[modelSize * inputSize];

    for (int i = 0; i < inputSize; ++i) {
        for (int j = 0; j < modelSize; ++j) {

        }
    }
}

int get_random_example(float *inputs) {
    assert(inputs && *inputs);

    srand((unsigned int) time(NULL));
    const int size = (sizeof(inputs) / sizeof(*inputs));
    return rand() % size;
}

template<typename T>
T two_dim_get(const T *array, int width, int i, int j) {
    return array[width * j + i];
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}