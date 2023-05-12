#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "../common/include/colors.h"
#include "../common/include/utils.h"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#ifdef __CUDACC__
__global__ void matrix_mul_kernel(float** Md, float** Nd, float** Pd, int n, int p, int q) {
    // Chaque thread calcule toutes les multiplications utilisant l'élément Nd[tx][ty]
    int tx = (blockIdx.x*blockDim.x) + threadIdx.x; // Indice de colonne
    int ty = (blockIdx.y*blockDim.y) + threadIdx.y; // Indice de ligne

    if (tx >= p || ty >= q) {
        return;
    }

    for (int i = 0; i < n; i++) {
        atomicAdd(&(Pd[i][ty]), Md[i][tx]*Nd[tx][ty]);
        // P[i][ty] += P[i][tx] * N[tx][ty]
    }
}


void matrix_multiplication_device(float** m1, float** m2, float** result, int n, int p, int q) {
    // Traitement
    dim3 gridSize(i_div_up(p, BLOCKSIZE_x), i_div_up(q, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y);

    matrix_mul_kernel<<<gridSize, blockSize>>>(m1, m2, result, n, p, q);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif


void matrix_multiplication_host(float** m1, float** m2, float** result, int n, int p, int q) {
    for (int i=0; i < n; i++) {
        for (int j=0; j < q; j++) {
            result[i][j] = 0.;
            for (int k=0; k < p; k++) {
                result[i][j] += m1[i][k] * m2[k][j];
            }
        }
    }
}


void matrix_multiplication(float** m1, float** m2, float** result, int n, int p, int q, bool use_cuda) {
    #ifdef __CUDACC__
    if (use_cuda) {
        matrix_multiplication_device(m1, m2, result, n, p, q);
    } else {
        matrix_multiplication_host(m1, m2, result, n, p, q);
    }
    #else
    matrix_multiplication_host(m1, m2, result, n, p, q);
    #endif
}