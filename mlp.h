#ifndef MACHINE_LEARNING_MLP_H
#define MACHINE_LEARNING_MLP_H

#include <cmath>
#include <iostream>
#include <cassert>

void init();

double activation(double &x);

double get_random_double(double min, double max);

double ***get_random_model(int *modelStruct, int modelStructSize, int inputsSize);

template<typename T>
int get_random_example_pos(T *examples, int examplesSize, int inputSize);

template<typename T>
T two_dim_get(T *&array, int &width, int &x, int &y);


template<typename T>
T *get_example_at(T *examples, int inputSize, int pos);

double **mlp_regression_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                     int inputsSize);

double **mlp_classification_feed_forward(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                         int inputsSize);

void mlp_update_weight(double &weight, double &learningRate, double &output, double &error);

void mlp_regression_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                   int inputsSize, double **outputs, double *desiredOutputs, double &learningRate);

void mlp_classification_back_propagate(double ***model, int *modelStruct, int modelStructSize, double *inputs,
                                       int inputsSize, double **outputs, int *desiredOutputs, double &learningRate);

void mlp_classification_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
                            int *desiredOutput, double learningRate);

void mlp_regression_fit(double ***model, int *modelStruct, int modelStructSize, double *inputs, int inputsSize,
                        double *desiredOutput, double learningRate);

void
mlp_classification_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                         int inputsSize,
                         int **desiredOutputs, double learningRate, int epochs);

void
mlp_regression_train(double ***model, int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                     int inputsSize,
                     double **desiredOutputs, double learningRate, int epochs);

double ***mlp_classification_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                                          int inputsSize,
                                          int **desiredOutputs, double learningRate, int epochs);

double ***mlp_regression_create_model(int *modelStruct, int modelStructSize, double *examples, int examplesSize,
                                      int inputsSize,
                                      double **desiredOutputs, double learningRate, int epochs);

}


#endif //MACHINE_LEARNING_MLP_H
