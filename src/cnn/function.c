#include <stdio.h>
#include <math.h>
#include <float.h>

#include "include/function.h"


float max_float(float a, float b) {
    return a < b ? b:a;
}

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
    return max_float(0, x);
}

float relu_derivative(float x) {
    if (x > 0)
        return 1;
    return 0;
}

float tanh_(float x) {
    return tanh(x);
}

float tanh_derivative(float x) {
    float a = tanh(x);
    return 1 - a*a;
}

void apply_softmax_input(float ***input, int depth, int rows, int columns) {
    float m = FLT_MIN;
    float sum=0;
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                m = max_float(m, input[i][j][k]);
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

void choose_apply_function_matrix(int activation, float*** input, int depth, int dim) {
    if (activation == RELU) {
        apply_function_input(relu, input, depth, dim, dim);
    } else if (activation == SIGMOID) {
        apply_function_input(sigmoid, input, depth, dim, dim);
    } else if (activation == SOFTMAX) {
        apply_softmax_input(input, depth, dim, dim);
    } else if (activation == TANH) {
        apply_function_input(tanh_, input, depth, dim, dim);
    } else {
        printf("Erreur, fonction d'activation inconnue (choose_apply_function_matrix): %d\n", activation);
    }
}

void choose_apply_function_vector(int activation, float*** input, int dim) {
    if (activation == RELU) {
        apply_function_input(relu, input, 1, 1, dim);
    } else if (activation == SIGMOID) {
        apply_function_input(sigmoid, input, 1, 1, dim);
    } else if (activation == SOFTMAX) {
        apply_softmax_input(input, 1, 1, dim);
    } else if (activation == TANH) {
        apply_function_input(tanh_, input, 1, 1, dim);
    } else {
        printf("Erreur, fonction d'activation inconnue (choose_apply_function_vector): %d\n", activation);
    }
}

ptr get_function_activation(int activation) {
    if (activation == RELU) {
        return &relu;
    }
    if (activation == -RELU) {
        return &relu_derivative;
    }
    if (activation == -IDENTITY) {
        return &identity_derivative;
    }
    if (activation == IDENTITY) {
        return &identity;
    }
    if (activation == SIGMOID) {
        return &sigmoid;
    }
    if (activation == -SIGMOID) {
        return &sigmoid_derivative;
    }
    if (activation == SOFTMAX) {
        printf("Erreur, impossible de renvoyer la fonction softmax\n");
        return NULL;
    }
    if (activation == -SOFTMAX) {
        printf("Erreur, impossible de renvoyer la dérivée de la fonction softmax\n");
        return NULL;
    }
    if (activation == TANH) {
        return &tanh_;
    }
    if (activation == -TANH) {
        return &tanh_derivative;
    }
    printf("Erreur, fonction d'activation inconnue (choose_apply_function_vector): %d\n", activation);
    return NULL;
}
