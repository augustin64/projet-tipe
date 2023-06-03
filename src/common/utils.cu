#include <stdbool.h>
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


int i_div_up(int a, int b) { // Partie entière supérieure de a/b
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

#ifdef __CUDACC__
extern "C"
#endif
bool cuda_setup(bool verbose) {
    #ifdef __CUDACC__
    int nDevices;
    int selected_device = 0;
    cudaDeviceProp selected_prop;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0) { // I've seen weird issues when there is no GPU at all
        if (verbose) {
            printf("Pas d'utilisation du GPU\n\n");
        }
        return false;
    }

    if (verbose) {
        printf("GPUs disponibles:\n");
    }

    cudaGetDeviceProperties(&selected_prop, selected_device);

    for (int i=0; i < nDevices; i++) {
        cudaGetDeviceProperties(&prop, i);

        if (verbose) {
            printf(" - %s\n\t - Compute Capability: %d.%d\n\t - Memory available: ", prop.name, prop.major, prop.minor);
            printf_memory(prop.totalGlobalMem);
            printf("\n\t - Shared Memory per block: ");
            printf_memory(prop.sharedMemPerBlock);
            printf("\n\n");
        }

        if (prop.clockRate*prop.multiProcessorCount >= selected_prop.clockRate*selected_prop.multiProcessorCount) { // This criteria approximately matches the best device
            selected_prop = prop;
            selected_device = i;
        }
    }

    cudaSetDevice(selected_device); // Select the best device for computation
    if (verbose) {
        printf("Utilisation du GPU: " BLUE "%s" RESET "\n\n", selected_prop.name);
    }

    if (BLOCKSIZE_x*BLOCKSIZE_y*BLOCKSIZE_z > prop.maxThreadsPerBlock) {
        printf_error((char*)"La taille de bloc sélectionnée est trop grande.\n");
        printf("\tMaximum accepté: %d\n", selected_prop.maxThreadsPerBlock);
        exit(1);
    }
    if (selected_prop.sharedMemPerBlock != MEMORY_BLOCK) { // C'est un warning, on l'affiche dans tous les cas
        printf_warning((char*)"La taille des blocs mémoire du GPU et celle utilisée dans le code diffèrent.\n");
        printf("\tCela peut mener à une utilisation supplémentaire de VRAM.\n");
        printf("\tChanger MEMORY_BLOCK à %ld dans src/include/memory_management.h\n", selected_prop.sharedMemPerBlock);
    }
    return true;
    #else
    if (verbose) {
        printf("Pas d'utilisation du GPU\n\n");
    }

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