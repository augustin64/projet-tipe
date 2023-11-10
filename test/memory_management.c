#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "include/test.h"

#include "../src/common/include/memory_management.h"
#include "../src/common/include/colors.h"

#define N 350

int main() {
    _TEST_PRESENTATION("Memory management (C part)")

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