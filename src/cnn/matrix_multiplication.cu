#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "../common/include/colors.h"
#include "../common/include/utils.h"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#ifdef __CUDACC__
__global__ void matrix_mul_kernel(float** M, float** N, float** P, int n, int p, int q) {
    // Ce fil calcule toutes les multiplications utilisant l'élément N[idx][idy]
    int idx = (blockIdx.x*blockDim.x) + threadIdx.x; // Indice de colonne
    int idy = (blockIdx.y*blockDim.y) + threadIdx.y; // Indice de ligne

    if (idx >= p || idy >= q) {
        return; // On vérifie que l'on est bien à un emplacement valide
    }

    for (int i = 0; i < n; i++) {
        atomicAdd(&(P[i][idy]), M[i][idx]*N[idx][idy]);
        // P[i][idy] += M[i][idx] * N[idx][idy]
    }
}

void matrix_multiplication_device(float** m1, float** m2, float** result, int n, int p, int q) {
    // On découpe la tâche en un certain nombre de blocs,
    // la taille d'un bloc étant limitée par CUDA à 1024
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