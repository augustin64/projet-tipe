#include <stdio.h>

#include "../include/colors.h"
#include "include/convolution.h"
#include "include/make.h"


void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim) {
    // input[output_depth][output_dim+size-1][output_dim+size-1]
    // output[output_depth][output_dim][output_dim]
    float sum;
    int n = size*size;
    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_dim; j++) {
            for (int k=0; k < output_dim; k++) {
                sum = 0.;
                for (int a=0; a < size; a++) {
                    for (int b=0; b < size; b++) {
                        sum += input[i][size*j +a][size*k +b];
                    }
                }
                output[i][j][k] = sum/(float)n;
            }
        }
    }
}

void make_dense(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output) {
    // input[size_input]
    // output[size_output]
    float f;

    for (int i=0; i < size_output; i++) {
        f = kernel->bias[i];
        for (int j=0; j < size_input; j++) {
            f += kernel->weights[j][i]*input[j];
        }
        output[i] = f;
    }
}

void make_dense_linearised(Kernel_nn* kernel, float*** input, float* output, int depth_input, int dim_input, int size_output) {
    // input[depth_input][dim_input][dim_input]
    // output[size_output]
    float f;

    for (int l=0; l < size_output; l++) {
        f = 0;
        for (int i=0; i < depth_input; i++) {
            for (int j=0; j < dim_input; j++) {
                for (int k=0; k < dim_input; k++) {
                    f += input[i][j][k]*kernel->weights[k + j*dim_input + i*depth_input][l];
                }
            }
        }
        output[l] = f;
    }
}