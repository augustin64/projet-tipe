#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "../include/colors.h"
#include "../include/utils.h"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

#ifdef __CUDACC__
__global__ void matrix_mul_kernel(float* Md, float* Nd, float* Pd, int n, int p, int q, size_t pitch_m, size_t pitch_n, size_t pitch_p) {
    // Chaque thread calcule toutes les multiplications utilisant l'élément Nd[tx][ty]
    int tx = (blockIdx.x*blockDim.x) + threadIdx.x; // Indice de colonne
    int ty = (blockIdx.y*blockDim.y) + threadIdx.y; // Indice de ligne

    if (tx >= p || ty >= q) {
        return;
    }

    // Pvalue stores the Pd element that is computed by the thread
    float* M_offset;
    float* P_offset;
    float* N_offset = (float *)((char*)Nd + tx * pitch_n);
    float Nxy = N_offset[ty]; // N[tx][ty]

    for (int i = 0; i < n; i++) {
        M_offset = (float *)((char*)Md + i * pitch_m);
        P_offset = (float*)((char*)Pd + i * pitch_p); // P[i], pitch_p est un décalage en bytes
        atomicAdd(&P_offset[ty], M_offset[tx] * Nxy); // P[i][ty] += P[i][tx] * N[tx][ty]
    }
}


void matrix_multiplication_device(float** m1, float** m2, float** result, int n, int p, int q) {
    // Préparation des matrices
    size_t pitch_m1_dev;
    size_t pitch_m2_dev;
    size_t pitch_result_dev;
    float* m1_dev;
    float* m2_dev;
    float* result_dev;

    gpuErrchk( cudaMallocPitch((void**)&m1_dev, &pitch_m1_dev, p * sizeof(float), n));
    for (int i=0; i < n; i++) {
        gpuErrchk( cudaMemcpy((void*)((char*)m1_dev + i*pitch_m1_dev), (const void*)&(m1[i][0]), p*sizeof(float), cudaMemcpyHostToDevice));
    }

    gpuErrchk( cudaMallocPitch((void**)&m2_dev, &pitch_m2_dev, q * sizeof(float), p));
    for (int i=0; i < p; i++) {
        gpuErrchk( cudaMemcpy((void*)((char*)m2_dev + i*pitch_m2_dev), (const void*)&(m2[i][0]), q*sizeof(float), cudaMemcpyHostToDevice));
    }

    gpuErrchk( cudaMallocPitch((void**)&result_dev, &pitch_result_dev, q * sizeof(float), n));
    gpuErrchk( cudaMemset(result_dev, 0, pitch_result_dev*n));

    // Traitement
    dim3 gridSize(i_div_up(p, BLOCKSIZE_x), i_div_up(q, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y);

    matrix_mul_kernel<<<gridSize, blockSize>>>(m1_dev, m2_dev, result_dev, n, p, q, pitch_m1_dev, pitch_m2_dev, pitch_result_dev);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Post-traitement
    for (int i=0; i < n; i++) {
        gpuErrchk( cudaMemcpy((void*)&(result[i][0]), (const void*)((char*)result_dev + i*pitch_result_dev), sizeof(float)*q, cudaMemcpyDeviceToHost));
    }

    gpuErrchk( cudaFree(result_dev) );
    gpuErrchk( cudaFree(m1_dev) );
    gpuErrchk( cudaFree(m2_dev) );
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