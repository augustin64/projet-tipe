#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "include/struct.h"
#include "../include/utils.h"


#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 8
#define BLOCKSIZE_z 8


void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    // c'est le kernel de input
    // input[kernel->rows][kernel_k_size + output_dim-1][kernel_k_size + output_dim-1]
    // output[kernel->columns][output_dim][output_dim]
    float f;

    for (int i=0; i < kernel->columns; i++) { // filtre
        for (int j=0; j < output_dim; j++) { // ligne de sortie
            for (int k=0; k < output_dim; k++) { // colonne de sortie
                f = kernel->bias[i][j][k];
                for (int a=0; a < kernel->rows; a++) { // Canal de couleur
                    for (int b=0; b < kernel->k_size; b++) { // ligne du filtre
                        for (int c=0; c < kernel->k_size; c++) { // colonne du filtre
                            f += kernel->weights[a][i][b][c]*input[a][j+b][k+c];
                        }
                    }
                }
                output[i][j][k] = f;
            }
        }
    }
}

#ifdef __CUDACC__

__global__ void make_convolution_kernel(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < kernel->columns
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < min(output_dim, k_size)
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < min(output_dim, k_size)

    if (idx >= kernel->columns || idy >= output_dim || idz >= output_dim) {
        return;
    }

    float f = kernel->bias[idx][idy][idz];

    for (int a=0; a < kernel->rows; a++) {
        for (int b=0; b < kernel->k_size; b++) {
            for (int c=0; c < kernel->k_size; c++) {
                f += kernel->weights[a][idx][b][c]*input[a][idy+b][idz+c];
            }
        }
    }

    output[idx][idy][idz] = f;
}

void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    // Make computation
    dim3 gridSize(i_div_up(kernel->columns, BLOCKSIZE_x), i_div_up(output_dim, BLOCKSIZE_y), i_div_up(output_dim, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_convolution_kernel<<<gridSize, blockSize>>>(kernel, input, output, output_dim);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    #ifndef __CUDACC__
    make_convolution_cpu(kernel, input, output, output_dim);
    #else
    make_convolution_device(kernel, input, output, output_dim);
    #endif
}