#include <stdlib.h>
#include <stdio.h>
#ifdef USE_CUDA
    #ifndef __CUDACC__
        #include "cuda_runtime.h"
    #endif
#endif

#include "include/memory_management.h"
#include "include/colors.h"

#include "include/utils.h"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 8
#define BLOCKSIZE_z 8


#ifndef __CUDACC__
int min(int a, int b) {
    return a<b?a:b;
}

int max(int a, int b) {
    return a > b ? a : b;
}
#endif

int not_outside(int x, int y, int lower_bound, int upper_bound) {
    return !(x < lower_bound || y < lower_bound || x >= upper_bound || y>= upper_bound);
}

int i_div_up(int a, int b) { // Partie entière supérieure de a/b
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

#ifdef __CUDACC__
extern "C"
#endif
bool check_cuda_compatibility() {
    #ifdef __CUDACC__
    int nDevices;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        printf("Pas d'utilisation du GPU\n\n");
        return false;
    }

    printf("GPUs disponibles:\n");

    for (int i=0; i < nDevices; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf(" - %s\n\t - Compute Capability: %d.%d\n\t - Memory available: ", prop.name, prop.major, prop.minor);
        printf_memory(prop.totalGlobalMem);
        printf("\n\t - Shared Memory per block: ");
        printf_memory(prop.sharedMemPerBlock);
        printf("\n\n");
    }

    cudaGetDeviceProperties(&prop, 0);
    printf("Utilisation du GPU: " BLUE "%s" RESET "\n\n", prop.name);

    if (prop.sharedMemPerBlock != MEMORY_BLOCK) {
        printf_warning((char*)"La taille des blocs mémoire du GPU et celle utilisée dans le code diffèrent.\n");
        printf("\tCela peut mener à une utilisation supplémentaire de VRAM.\n");
        printf("\tChanger MEMORY_BLOCK à %ld dans src/include/memory_management.h\n", prop.sharedMemPerBlock);
    }
    return true;
    #else
    printf("Pas d'utilisation du GPU\n\n");
    return false;
    #endif
}

#ifdef __CUDACC__
__global__ void copy_3d_array_kernel(float*** source, float*** dest, int dimension1, int dimension2, int dimension3) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < dimension1
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < dimension2
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < dimension3

    if (idx >= dimension1 || idy >= dimension2 || idz >= dimension3) {
        return;
    }

    dest[idx][idy][idz] = source[idx][idy][idz];
}

void copy_3d_array(float*** source, float*** dest, int dimension1, int dimension2, int dimension3) {
    dim3 gridSize(i_div_up(dimension1, BLOCKSIZE_x), i_div_up(dimension2, BLOCKSIZE_y), i_div_up(dimension3, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    copy_3d_array_kernel<<<gridSize, blockSize>>>(source, dest, dimension1, dimension2, dimension3);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#else
void copy_3d_array(float*** source, float*** dest, int dimension1, int dimension2, int dimension3) {
    for (int i=0; i < dimension1; i++) {
        for (int j=0; j < dimension2; j++) {
            for (int k=0; k < dimension3; k++) {
                dest[i][j][k] = source[i][j][k];
            }
        }
    }
}
#endif

#ifdef __CUDACC__
__global__ void reset_3d_array_kernel(float*** dest, int dimension1, int dimension2, int dimension3) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < dimension1
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < dimension2
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < dimension3

    if (idx >= dimension1 || idy >= dimension2 || idz >= dimension3) {
        return;
    }

    dest[idx][idy][idz] = 0.;
}

extern "C"
void reset_3d_array(float*** dest, int dimension1, int dimension2, int dimension3) {
    dim3 gridSize(i_div_up(dimension1, BLOCKSIZE_x), i_div_up(dimension2, BLOCKSIZE_y), i_div_up(dimension3, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    reset_3d_array_kernel<<<gridSize, blockSize>>>(dest, dimension1, dimension2, dimension3);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#else
void reset_3d_array(float*** dest, int dimension1, int dimension2, int dimension3) {
    for (int i=0; i < dimension1; i++) {
        for (int j=0; j < dimension2; j++) {
            for (int k=0; k < dimension3; k++) {
                dest[i][j][k] = 0.;
            }
        }
    }
}
#endif