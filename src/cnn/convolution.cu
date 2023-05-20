#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "include/struct.h"
#include "../common/include/utils.h"

#include "include/config.h"


void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding) {
    // c'est le kernel de input
    // input[kernel->rows][kernel_k_size + output_width-1][kernel_k_size + output_width-1]
    // output[kernel->columns][output_width][output_width]
    
    int k_columns = kernel->columns;
    int k_rows = kernel->rows;
    int max_move = kernel->k_size - padding;
    int input_width = output_width*stride - 2*padding + kernel->k_size - stride;
    float f;

    for (int i=0; i < k_columns; i++) { // filtre
        for (int j=0; j < output_width; j++) { // ligne de sortie
            for (int k=0; k < output_width; k++) { // colonne de sortie
                f = kernel->bias[i][j][k];
                for (int a=0; a < k_rows; a++) { // Canal de couleur
                    for (int b=-padding; b < max_move; b++) { // ligne du filtre
                        for (int c=-padding; c < max_move; c++) { // colonne du filtre
                            int x = (stride*j+b);
                            int y = (stride*k+c);
                            if (not_outside(x, y, 0, input_width)) {
                                f += kernel->weights[a][i][b+padding][c+padding]*input[a][x][y];
                            }
                        }
                    }
                }
                output[i][j][k] = f;
            }
        }
    }
}

#ifdef __CUDACC__

__global__ void make_convolution_kernel(float**** weights, float*** bias, int k_size, int rows, int columns, float*** input, float*** output, int output_width, int stride, int padding) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < kernel->columns
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < min(output_width, k_size)
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < min(output_width, k_size)
    int max_move = k_size - padding;
    int input_width = output_width*stride - 2*padding + k_size - stride;

    if (idx >= columns || idy >= output_width || idz >= output_width) {
        return;
    }

    float f = bias[idx][idy][idz];

    for (int a=0; a < rows; a++) {
        for (int b=-padding; b < max_move; b++) {
            for (int c=-padding; c < max_move; c++) {
                int idy_2 = idy*stride+b;
                int idz_2 = idz*stride+c;
                if (not_outside(idy_2, idz_2, 0, input_width)) {
                    f += weights[a][idx][b+padding][c+padding]*input[a][idy_2][idz_2];
                }
            }
        }
    }

    output[idx][idy][idz] = f;
}

void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding) {
    // Make computation
    dim3 gridSize(i_div_up(kernel->columns, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    // We can't pass `kernel` directly to the CUDA kernel function
    // as it will create a 'misaligned adress' error
    make_convolution_kernel<<<gridSize, blockSize>>>(kernel->weights, kernel->bias, kernel->k_size, kernel->rows, kernel->columns, input, output, output_width, stride, padding);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

#ifdef __CUDACC__
extern "C"
#endif
void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_width, int stride, int padding) {
    #ifndef __CUDACC__
    make_convolution_cpu(kernel, input, output, output_width, stride, padding);
    #else
    make_convolution_device(kernel, input, output, output_width, stride, padding);
    #endif
}