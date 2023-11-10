#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "include/test.h"

#include "../src/common/include/memory_management.h"
#include "../src/common/include/colors.h"
#include "../src/common/include/utils.h"

#define N 350

__global__ void check_access(int* array, int range) {
    for (int i=0; i < range; i++) {
        array[i]++;
    }
}



int main() {
    _TEST_PRESENTATION("Memory management (Cuda part)")

    printf("Checking CUDA compatibility.\n");
    bool cuda_compatible = cuda_setup(true);
    if (!cuda_compatible) {
        printf(RED "CUDA not compatible, skipping tests.\n" RESET);
        return 0;
    }
    printf(GREEN "OK\n" RESET);

    int mem_used;
    int blocks_used;
    // We pollute a little bit the memory before the tests
    int* pointeurs[N];
    for (int i=1; i < N; i++) {
        pointeurs[i] = (int*)nalloc(i, sizeof(int));
        for (int j=0; j < i; j++) {
            pointeurs[i][j] = i;
        }
    }

    _TEST_ASSERT(true, "Pollution de la mémoire");

    // We test in a first place that one simple allocation works as expected
    mem_used = get_memory_distinct_allocations();
    blocks_used = get_memory_blocks_number();
    void* ptr = nalloc(15, 1);
    
    _TEST_ASSERT((get_memory_distinct_allocations() <= mem_used+1), "Un seul bloc mémoire alloué par allocation (max)");
    
    gree(ptr, false);

    _TEST_ASSERT((get_memory_blocks_number() == blocks_used), "Libération partielle de la mémoire");

    /* On lance des kernels de taille 1 ce qui est à la fois itératif et synchrone
    * Donc un peu contraire à CUDA mais l'objectif est de pouvoir débugger facilement */
    dim3 gridSize(1, 1, 1);
    dim3 blockSize(1, 1, 1);

    for (int i=1; i < N; i++) {
        check_access<<<gridSize, blockSize>>>(pointeurs[i], i);
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    _TEST_ASSERT(true, "accès CUDA à la mémoire")
    // We test that we do not use too much blocks
    blocks_used = get_memory_blocks_number();
    void* ptr1 = nalloc(-1+MEMORY_BLOCK/2, 1);
    void* ptr2 = nalloc(-1+MEMORY_BLOCK/2, 1);

    _TEST_ASSERT((get_memory_blocks_number() <= blocks_used +1), "Taille d'allocation de deux demi-blocs");


    for (int i=1; i < N; i++) {
        for (int j=0; j < i; j++) {
            // We test that the memory does not overlap itself
            assert(pointeurs[i][j] == i);
        }
        gree(pointeurs[i], false);
    }

    gree(ptr1, false);
    gree(ptr2, false);
    _TEST_ASSERT((get_memory_distinct_allocations() == 0 && get_memory_blocks_number() == 0), "Libération totale de la mémoire");

    return 0;
}