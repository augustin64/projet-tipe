#include <stdio.h>
#include "make.h"

void make_convolution(float*** input, Kernel_cnn* kernel, float*** output, int output_dim) {
    //NOT FINISHED, MISS CONDITIONS ON THE CONVOLUTION
    float f;
    int i, j, k, a, b, c, n=kernel->k_size;
    for (i=0; i<kernel->columns; i++) {
        for (j=0; j<output_dim; j++) {
            for (k=0; k<output_dim; k++) {
                f = kernel->bias[i][j][k];
                for (a=0; a<kernel->rows; a++) {
                    for (b=0; b<n; b++) {
                        for (c=0; c<n; c++) {
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
    //NOT FINISHED, MISS CONDITIONS ON THE POOLING
    float average;
    int i, j, k, a, b, n=size*size;
    for (i=0; i<output_depth; i++) {
        for (j=0; j<output_dim; j++) {
            for (k=0; k<output_dim; k++) {
                average = 0.;
                for (a=0; a<size; a++) {
                    for (b=0; b<size; b++) {
                        average += input[i][2*j +a][2*k +b];
                    }
                }
                output[i][j][k] = average;
            }
        }
    }
}

void make_average_pooling_flattened(float*** input, float* output, int size, int input_depth, int input_dim) {
    if ((input_depth*input_dim*input_dim) % (size*size) != 0) {
        printf("Erreur, deux layers non compatibles avec un average pooling flattened");
        return;
    }
    float average;
    int i, j, k, a, b, n=size*size, cpt=0;
    int output_dim = input_dim - 2*(size/2);
    for (i=0; i<input_depth; i++) {
        for (j=0; j<output_dim; j++) {
            for (k=0; k<output_dim; k++) {
                average = 0.;
                for (a=0; a<size; a++) {
                    for (b=0; b<size; b++) {
                        average += input[i][2*j +a][2*k +b];
                    }
                }
                output[cpt] = average;
                cpt++;
            }
        }
    }
}

void make_fully_connected(float* input, Kernel_nn* kernel, float* output, int size_input, int size_output) {
    int i, j, k;
    float f;
    for (i=0; i<size_output; i++) {
        f = kernel->bias[i];
        for (j=0; j<size_input; j++) {
            f += kernel->weights[i][j]*input[j];
        }
        output[i] = f;
    }
}