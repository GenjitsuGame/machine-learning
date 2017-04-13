#include <iostream>
#include <cassert>



int sign(float input) {
    if (input >= 0) return 1;
    else return -1;
}

void linear_regression(float *model, float *input, int epochs) {
    assert(epochs > 0 && input && model);

}

int get_random_example(float *inputs) {
    assert(inputs && *inputs);

    srand ((unsigned int) time(NULL));
    const int size = (sizeof(inputs)/sizeof(*inputs));
    return rand() % size;
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}