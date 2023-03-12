#include <stdio.h>
#include <math.h>
#include <float.h>

#include "../include/colors.h"

#include "include/function.h"
#define BOUND_RELU 15

float identity(float x) {
    return x;
}

float identity_derivative(float x) {
    (void)x;
    return 1;
}


float sigmoid(float x) {
    return 1/(1 + exp(-x));
}

float sigmoid_derivative(float x) {
    float tmp = exp(-x);
    return tmp/((1+tmp)*(1+tmp));
}


float relu(float x) {
    return fmaxf(0, fminf(x, BOUND_RELU));
}

float relu_derivative(float x) {
    if (x > 0)
        return 1;
    return 0;
}


float leaky_relu(float x) {
    if (x>0)
        return fminf(x, BOUND_RELU);
    return x*LEAKER;
}

float leaky_relu_derivative(float x) {
    if (x > 0)
        return 1;
    return LEAKER;
}


float tanh_(float x) {
    return tanh(x);
}

float tanh_derivative(float x) {
    float a = tanh(x);
    return 1 - a*a;
}


void apply_softmax_input(float ***input, int depth, int rows, int columns) {
    float m = -FLT_MAX;
    float sum=0;
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                m = fmaxf(m, input[i][j][k]);
            }
        }
    }
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                input[i][j][k] = exp(m-input[i][j][k]);
                sum += input[i][j][k];
            }
        }
    }
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                input[i][j][k] = input[i][j][k]/sum;
            }
        }
    }
}


void apply_function_input(float (*f)(float), float*** input, int depth, int rows, int columns) {
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                input[i][j][k] = (*f)(input[i][j][k]);
            }
        }
    }
}

void apply_function_to_matrix(int activation, float*** input, int depth, int dim) {
    if (activation == SOFTMAX) {
        return apply_softmax_input(input, depth, dim, dim);
    }
    if (activation >= 1) { // Exclude negative values (derivative)
        ptr f = get_activation_function(activation);
        return apply_function_input(f, input, depth, dim, dim);
    }
    printf_error("fonction d'activation inconnue (apply_function_to_matrix): ");
    printf("%d\n", activation);
}


void apply_function_to_vector(int activation, float*** input, int dim) {
    if (activation == SOFTMAX) {
        return apply_softmax_input(input, 1, 1, dim);
    }
    if (activation >= 1) { // Exclude negative values (derivative)
        ptr f = get_activation_function(activation);
        return apply_function_input(f, input, 1, 1, dim);
    }
    printf_error("fonction d'activation inconnue (apply_function_to_vector): ");
    printf("%d\n", activation);
}


ptr get_activation_function(int activation) {
    if (activation == RELU) {
        return &relu;
    }
    if (activation == -RELU) {
        return &relu_derivative;
    }

    if (activation == IDENTITY) {
        return &identity;
    }
    if (activation == -IDENTITY) {
        return &identity_derivative;
    }

    if (activation == SIGMOID) {
        return &sigmoid;
    }
    if (activation == -SIGMOID) {
        return &sigmoid_derivative;
    }

    if (activation == SOFTMAX) {
        printf_error("impossible de renvoyer la fonction softmax\n");
        return NULL;
    }
    if (activation == -SOFTMAX) {
        printf_error("impossible de renvoyer la dérivée de la fonction softmax\n");
        return NULL;
    }

    if (activation == TANH) {
        return &tanh_;
    }
    if (activation == -TANH) {
        return &tanh_derivative;
    }

    if (activation == LEAKY_RELU) {
        return &leaky_relu;
    }
    if (activation == -LEAKY_RELU) {
        return &leaky_relu_derivative;
    }
    printf_error("fonction d'activation inconnue (get_activation_function): ");
    printf("%d\n", activation);
    return NULL;
}
