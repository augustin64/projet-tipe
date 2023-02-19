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
