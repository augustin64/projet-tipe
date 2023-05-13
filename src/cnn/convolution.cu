#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "include/struct.h"
#include "../common/include/utils.h"

#include "include/config.h"


int convolution_not_outside(int x, int y, int lower_bound, int upper_bound) {
    // On renvoie true si et seulement si _ et _:
    // lower_bound <= x < upper_bound
    // lower_bound <= y < upper_bound
    
    return !(x < lower_bound || y < lower_bound || x >= upper_bound || y>= upper_bound);
}

void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding) {
    // c'est le kernel de input
    // input[kernel->rows][kernel_k_size + output_dim-1][kernel_k_size + output_dim-1]
    // output[kernel->columns][output_dim][output_dim]
    
    int k_columns = kernel->columns;
    int k_rows = kernel->rows;
    int max_move = kernel->k_size - padding;
    int input_dim = output_dim*stride - 2*padding + kernel->k_size - stride;
    float f;

    for (int i=0; i < k_columns; i++) { // filtre
        for (int j=0; j < output_dim; j++) { // ligne de sortie
            for (int k=0; k < output_dim; k++) { // colonne de sortie
                f = kernel->bias[i][j][k];
                for (int a=0; a < k_rows; a++) { // Canal de couleur
                    for (int b=-padding; b < max_move; b++) { // ligne du filtre
                        for (int c=-padding; c < max_move; c++) { // colonne du filtre
                            int x = (stride*j+b);
                            int y = (stride*k+c);
                            if (convolution_not_outside(x, y, 0, input_dim)) {
                                f += kernel->weights[a][i][b][c]*input[a][stride*j+b][stride*k+c];
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

__global__ void make_convolution_kernel(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < kernel->columns
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < min(output_dim, k_size)
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < min(output_dim, k_size)
    int max_move = kernel->k_size - padding;
    int input_dim = output_dim*stride - 2*padding + kernel->k_size - stride;

    if (idx >= kernel->columns || idy >= output_dim || idz >= output_dim) {
        return;
    }

    float f = kernel->bias[idx][idy][idz];

    for (int a=0; a < kernel->rows; a++) {
        for (int b=-padding; b < max_move; b++) {
            for (int c=-padding; c < max_move; c++) {
                int idy_2 = idy*stride+b;
                int idz_2 = idz*stride+c;
                if (convolution_not_outside(idy_2, idz_2, 0, input_dim)) {
                    f += kernel->weights[a][idx][b][c]*input[a][idy_2][idz_2];
                }
            }
        }
    }

    output[idx][idy][idz] = f;
}

void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding) {
    // Make computation
    dim3 gridSize(i_div_up(kernel->columns, BLOCKSIZE_x), i_div_up(output_dim, BLOCKSIZE_y), i_div_up(output_dim, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_convolution_kernel<<<gridSize, blockSize>>>(kernel, input, output, output_dim, stride, padding);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim, int stride, int padding) {
    #ifndef __CUDACC__
    make_convolution_cpu(kernel, input, output, output_dim, stride, padding);
    #else
    make_convolution_device(kernel, input, output, output_dim, stride, padding);
    #endif
}