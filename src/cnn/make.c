#include <stdio.h>

#include "../colors.h"
#include "include/make.h"

void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    float f;
    int n = kernel->k_size;
    printf("max_input %dx%dx%d: %d \n", kernel->rows, n+output_dim -1, output_dim+n -1, n);
    for (int i=0; i < kernel->columns; i++) {
        for (int j=0; j < output_dim; j++) {
            for (int k=0; k < output_dim; k++) {
                f = kernel->bias[i][j][k];
                for (int a=0; a < kernel->rows; a++) {
                    for (int b=0; b < n; b++) {
                        for (int c=0; c < n; c++) {
                            f += kernel->w[a][i][b][c]*input[a][j+b][k+c];
                        }
                    }
                }
                output[i][j][k] = f/n; // Average
            }
        }
    }
}

void make_average_pooling(float*** input, float*** output, int size, int output_depth, int output_dim) {
    printf("%d -> %d \n", output_dim*size, output_dim);
    float average;
    int n = size*size;
    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_dim; j++) {
            for (int k=0; k < output_dim; k++) {
                average = 0.;
                for (int a=0; a < size; a++) {
                    for (int b=0; b < size; b++) {
                        average += input[i][size*j +a][size*k +b];
                    }
                }
                output[i][j][k] = average/n;
            }
        }
    }
}

void make_dense(Kernel_nn* kernel, float* input, float* output, int size_input, int size_output) {
    float f;
    for (int i=0; i < size_output; i++) {
        f = kernel->bias[i];
        for (int j=0; j < size_input; j++) {
            f += kernel->weights[i][j]*input[j];
        }
        output[i] = f;
    }
}

void make_dense_linearised(Kernel_nn* kernel, float*** input, float* output, int depth_input, int dim_input, int size_output) {
    int n = depth_input*dim_input*dim_input;
    printf("%dx%dx%d (%d) -> %d\n",depth_input, dim_input, dim_input, n, size_output);
    float f;
    for (int l=0; l<size_output; l++) {
        f = 0;
        for (int i=0; i<depth_input; i++) {
            for (int j=0; j<dim_input; j++) {
                for (int k=0; k<dim_input; k++) {
                    f += input[i][j][k]*kernel->weights[k + j*dim_input + i*depth_input][l];
                }
            }
        }
        output[l] = f/size_output;
    }
}