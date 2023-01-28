#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef USE_CUDA
    #include "cuda_runtime.h"
#endif
#include "include/utils.h"
#include "include/colors.h"


int i_div_up(int a, int b) { // Partie entière supérieure de a/b
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

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
        printf(" - %s\n", prop.name);
    }

    cudaGetDeviceProperties(&prop, 0);
    printf("Utilisation du GPU: " BLUE "%s" RESET " (Compute capability: %d.%d)\n\n", prop.name, prop.major, prop.minor);
    return true;
    #else
    printf("Pas d'utilisation du GPU\n\n");
    return false;
    #endif
}

#ifndef USE_CUDA

void* nalloc(size_t sz) {
    void* ptr = malloc(sz);
    return ptr;
}

void gree(void* ptr) {
    free(ptr);
}

#else

void* nalloc(size_t sz) {
    void* ptr;
    cudaMallocManaged(&ptr, sz, cudaMemAttachHost);
    return ptr;
}

void gree(void* ptr) {
    cudaFree(ptr);
}

#endif