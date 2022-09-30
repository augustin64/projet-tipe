#include <stdio.h>

#include "../colors.h"
#include "include/make.h"

void make_convolution(float*** input, Kernel_cnn* kernel, float*** output, int output_dim) {
    printf_warning("Appel de make_convolution, incomplet\n");
    float f;
    int n = kernel->k_size;
    for (int i=0; i < kernel->columns; i++) {
        for (int j=0; j < output_dim; j++) {
            for (int k=0; k < output_dim; k++) {
                f = kernel->bias[i][j][k];
                for (int a=0; a < kernel->rows; a++) {
                    for (int b=0; b < n; b++) {
                        for (int c=0; c < n; c++) {
                            f += kernel->w[a][i][b][c]*input[a][j+a][k+b];
                        }
                    }
                }
                output[i][j][k] = f;
            }
        }
    }
}

void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim) {
    // TODO, MISS CONDITIONS ON THE POOLING
    printf_warning("Appel de make_average_pooling, incomplet\n");
    float average;
    int n = size*size;
    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_dim; j++) {
            for (int k=0; k < output_dim; k++) {
                average = 0.;
                for (int a=0; a < size; a++) {
                    for (int b=0; b < size; b++) {
                        average += input[i][2*j +a][2*k +b];
                    }
                }
                output[i][j][k] = average/n;
            }
        }
    }
}

void make_fully_connected(float* input, Kernel_nn* kernel, float* output, int size_input, int size_output) {
    float f;
    for (int i=0; i < size_output; i++) {
        f = kernel->bias[i];
        for (int j=0; j < size_input; j++) {
            f += kernel->weights[i][j]*input[j];
        }
        output[i] = f;
    }
}