#include <iostream>
#include <cassert>

void init() {
    srand(static_cast <unsigned> (time(0)));
    std::cout << std::boolalpha;
}

int sign(float const &input) {
    if (input >= 0) return 1;
    else return -1;
}

float get_random_float(float const min, float const max) {
    return min + static_cast <float> (rand()) / ((RAND_MAX / (max - min)));
}

float **const get_random_model(int const *const inputs, int const inputsSize) {
    assert(inputs && *inputs && inputsSize > 0);

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

    return model;
}

template<typename T>
T *two_dim_get(T *const &array, int const &width, int const &x, int const &y) {
    return (array + width * y + x);
}

template<typename T>
void two_dim_set(T *const &array, int const &width, int const x, int const y, T e) {
    array[width * y + x] = e;
}

template<typename T>
int get_random_example_pos(T const *const examples, int const examplesSize, int const inputSize) {
    assert(examples);
    assert(examplesSize > 0);
    assert(inputSize > 0);

    return rand() % (examplesSize / inputSize);
}


template<typename T>
T *const *const get_example_at(T *const examples, int const inputSize, int const pos) {
    T **const example = new T *[inputSize];

    for (int i = 0; i < inputSize; ++i) {
        example[i] = two_dim_get(examples, inputSize, i, pos);
    }

    return example;
}

void
perceptron_learning_algorithm(float *const model, float const *const inputs, int const inputsSize,
                              int desiredOutput, float learningRate) {
    assert(inputs);
    assert(inputsSize > 0);
    assert(model);
    assert(learningRate > 0);

    for (int i = 0; i < inputsSize; ++i) {
        model[i] += learningRate * inputs[i] * desiredOutput;
    }
}

float linear_regression(float *model, int const modelSize, float const *const *const inputs, int const inputSize) {
    assert(model);
    assert(inputs);
    assert(modelSize == inputSize);

    float weightedSum = 0;

    for (int i = 0; i < inputSize; ++i) {
        weightedSum += *(inputs[i]) * model[i];
    }

    return weightedSum;
}

float *
create_linear_regression_model(const float *const examples, const int examplesSize, const int inputSize,
                               const int *const desiredOutputs, const int epochs, const float learningRate) {
    assert(examples);
    assert(desiredOutputs);
    assert(inputSize > 0);
    assert(epochs > 0);

    float *model = get_random_model(new int[1]{inputSize}, 1)[0];

    for (int i = 0; i < epochs; ++i) {
        const int pos = get_random_example_pos(examples, examplesSize, inputSize);
        float const *const *const example = get_example_at(examples, inputSize, pos);
        float const output = linear_regression(model, inputSize, example, inputSize);

    }

}

int linear_classification(float * const model, int const modelSize, float const *const *const inputs, const int inputSize) {
    assert(model && *model);
    assert(inputs && *inputs);

    return sign(linear_regression(model, modelSize, inputs, inputSize));
}

float *
create_linear_classification_model(const float *const examples, const int examplesSize, const int inputSize,
                                   const int *const desiredOutputs, const int epochs, const float learningRate) {
    assert(examples);
    assert(desiredOutputs);
    assert(inputSize > 0);
    assert(epochs > 0);

    float *const model = get_random_model(new int[1]{inputSize}, 1)[0];

    for (int i = 0; i < epochs; ++i) {
        const int pos = get_random_example_pos(examples, examplesSize, inputSize);
        float const *const *const example = get_example_at(examples, inputSize, pos);
        const int output = linear_classification(model, inputSize, example, inputSize);
        if (output != desiredOutputs[pos]) {
            perceptron_learning_algorithm(model, *example, inputSize, desiredOutputs[pos], learningRate);
        }
    }

    return model;
}

// TEST

const int classificationExamplesSize = 4;
const int classificationExampleSize = 2;
const float *const classificationExamples = new float[classificationExamplesSize * classificationExampleSize]{-1, -1,
                                                                                                              -1,
                                                                                                              1, 1, -1,
                                                                                                              1, 1};
const int *const classificationDesiredOutputs = new int[classificationExamplesSize]{-1, -1, 1, 1};

void test_get_example() {
    float const *const *const example = get_example_at(classificationExamples, classificationExampleSize, 1);
    assert(*example == (classificationExamples + 2));
}

void test_get_random_model() {
    const int layersSize = 3;
    const int *layersSizes = new int[layersSize]{4, 3, 5};

    float **const model = get_random_model(layersSizes, layersSize);
    for (int i = 0; i < layersSize; ++i) {
        for (int j = 0; j < layersSizes[i]; ++j) {
            assert(model[i][j]);
        }
        delete[] model[i];
    }

    delete[] model;
    delete[] layersSizes;
}

void test_linear_classification() {
    const int layersSize = 1;
    const int *layersSizes = new int[layersSize]{1};
    const int modelSize = 2;
    float *const model = new float[modelSize]{1, -1};
    const int desiredOutput = -1;
    const int examplePos = 1;
    float const *const *const example = get_example_at(classificationExamples, classificationExampleSize, examplePos);
    const int output = linear_classification(model, modelSize, example, classificationExampleSize);

    assert(desiredOutput == output);

    delete[] layersSizes;
    delete[] model;
}

void test_create_linear_classification_model() {
    int const epochs = 2000;
    int const desiredOutput = -1;
    int const examplePos = 1;
    float const learningRate = 0.01;
    float const *const *const example = get_example_at(classificationExamples, classificationExampleSize, examplePos);
    int const output = linear_classification(
            create_linear_classification_model(classificationExamples, classificationExamplesSize,
                                               classificationExampleSize, classificationDesiredOutputs, epochs,
                                               learningRate), classificationExampleSize, example,
            classificationExampleSize);

    assert(desiredOutput == output);
}

void test() {
    init();
    test_get_example();
    test_get_random_model();
    test_linear_classification();
    test_create_linear_classification_model();
}

int main() {
    test();
    return 0;
}