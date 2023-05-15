#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "../src/common/include/memory_management.h"
#include "../src/common/include/colors.h"

#define N 350

int main() {
    printf("Pollution de la mémoire\n");
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

    // We test in a first place that one simple allocation works as expected
    mem_used = get_memory_distinct_allocations();
    blocks_used = get_memory_blocks_number();
    void* ptr = nalloc(15, 1);
    if (! (get_memory_distinct_allocations() <= mem_used+1)) {
        printf_error((char*)"Plus d'un élément de mémoire alloué en une seule allocation\n");
        exit(1);
    }
    gree(ptr, false);
    if (! (get_memory_blocks_number() == blocks_used)) {
        printf_error((char*)"La mémoire n'a pas été libérée correctement\n");
        exit(1);
    }
    printf(GREEN "OK\n" RESET);



    printf("Allocation de deux demi-blocs\n");
    // We test that we do not use too much blocks
    blocks_used = get_memory_blocks_number();
    void* ptr1 = nalloc(-1+MEMORY_BLOCK/2, 1);
    void* ptr2 = nalloc(-1+MEMORY_BLOCK/2, 1);
    if (! (get_memory_blocks_number() <= blocks_used +1)) {
        printf_error((char*)"Trop de blocs ont été alloués par rapport à la mémoire nécessaire\n");
        exit(1);
    }
    printf(GREEN "OK\n" RESET);



    printf("Libération de la mémoire\n");
    for (int i=1; i < N; i++) {
        for (int j=0; j < i; j++) {
            // We test that the memory does not overlap itself
            assert(pointeurs[i][j] == i);
        }
        gree(pointeurs[i], false);
    }

    gree(ptr1, false);
    gree(ptr2, false);
    if (! (get_memory_distinct_allocations() == 0 && get_memory_blocks_number() == 0)) {
        printf_error((char*)"La mémoire n'a pas été libérée correctement\n");
        exit(1);
    }
    printf(GREEN "OK\n" RESET);
    return 0;
}