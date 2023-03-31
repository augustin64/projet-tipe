#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <pthread.h>

#include "include/memory_management.h"
#include "include/colors.h"
#include "include/utils.h"


Memory* memory = NULL;
pthread_mutex_t memory_lock = PTHREAD_MUTEX_INITIALIZER;


int get_distinct_allocations(Memory* mem) {
    if (!mem) {
        return 0;
    }
    return mem->nb_alloc + get_distinct_allocations(mem->next);
}


int get_length(Memory* mem) {
    if (!mem) {
        return 0;
    }
    return 1 + get_length(mem->next);
}


int get_memory_distinct_allocations() {
    return get_distinct_allocations(memory);
}

int get_memory_blocks_number() {
    return get_length(memory);
}

void print_memory_rec(Memory* mem) {
    if (!mem) {
        return;
    }
    printf("==== %u ====\n", mem->id);
    printf("plage d'addresses: %p-%p\n", mem->start, (void*)((intptr_t)mem->start +mem->size));
    printf("in-use: %ld/%ld\n", ((intptr_t)mem->cursor - (intptr_t)mem->start), mem->size);
    printf("allocations: %d\n\n", mem->nb_alloc);
    print_memory_rec(mem->next);
}


void print_memory() {
    printf(BLUE "==== MEMORY ====\n" RESET);
    print_memory_rec(memory);
}

Memory* create_memory_block(size_t size) {
    Memory* mem = (Memory*)malloc(sizeof(Memory));
    #ifdef __CUDACC__
    cudaMallocManaged(&(mem->start), size, cudaMemAttachHost);

    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    #else
    mem->start = malloc(size);
    #endif
    mem->cursor = mem->start;
    mem->size = size;
    mem->nb_alloc = 0;
    mem->next = NULL;
    mem->id = rand() %100000;
    
    return mem;
}


void* allocate_memory(int nb_elements, size_t size, Memory* mem) {
    /*
    * cursor_aligned pointe vers le premier emplacement qui pourrait être utilisé (de manière alignée).
    * en effet, la mémoire nécessite d'être alignée avec CUDA:
    * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses
    */
   void* aligned_cursor = mem->cursor;
    #ifdef __CUDACC__
        // Cela devrait être faisable avec opérateurs binaires directement, mais on préfèrera quelque chose de lisible et vérifiable
        if (((intptr_t)mem->cursor) %size != 0) {
            if (size == 2 || size == 4 || size == 8 || size == 16)
                aligned_cursor = (void*)(((intptr_t)mem->cursor) + (size - (((intptr_t)mem->cursor) %size)));
        }
    #endif
    // Si il y a suffisamment de mémoire disponible
    if (mem->size - ((intptr_t)aligned_cursor - (intptr_t)mem->start) >= nb_elements*size) {
        void* ptr = aligned_cursor;
        mem->cursor = (void*)((intptr_t)aligned_cursor + nb_elements*size); // On décale le curseur de la taille allouée
        mem->nb_alloc++;
        return ptr;
    } else {
        //printf("Mémoire disponible: %ld. Nécessaire: %ld\n", mem->size - ((intptr_t)mem->cursor - (intptr_t)mem->start), nb_elements*size);
        // Sinon on continue sur l'élément suivant de la liste
        if (!mem->next) {
            //! WARNING: May cause Infinite allocations when trying to allocate more than MEMORY_BLOCK size at once that is not naturally aligned (CUDA only)
            mem->next = create_memory_block(MEMORY_BLOCK < nb_elements*size ? nb_elements*size : MEMORY_BLOCK);
        }
        return allocate_memory(nb_elements, size, mem->next);
    }
}


Memory* free_memory(void* ptr, Memory* mem) {
    if (!mem) {
        printf_error((char*)"Le pointeur ");
        printf("%p a déjà été libéré ou n'a jamais été alloué\n", ptr);
        return mem;
    }
    if (mem->start <= ptr && ptr < (void*)((intptr_t)mem->start + mem->size)) {
        mem->nb_alloc--;
        // printf(GREEN "%p <= %p < %p\n" RESET, mem->start, ptr, (void*)((intptr_t)mem->start + mem->size));
        if (mem->nb_alloc == 0) {
            Memory* mem_next = mem->next;
            #ifdef __CUDACC__
            cudaFree(mem->start);
            #else
            free(mem->start);
            #endif
            free(mem);
            return mem_next;
        } else {
            return mem;
        }
    } else {
        mem->next = free_memory(ptr, mem->next);
        return mem;
    }
}


#ifdef __CUDACC__
extern "C"
#endif
void* nalloc(int nb_elements, size_t size) {
    #if defined(__CUDACC__) || defined(TEST_MEMORY_MANAGEMENT)
        pthread_mutex_lock(&memory_lock);
        if (!memory) {
            // We allocate a new memory block
            memory = create_memory_block(MEMORY_BLOCK < nb_elements*size ? nb_elements*size : MEMORY_BLOCK);
        }
        //printf("Distinct allocations: %d Blocks: %d\n", get_distinct_allocations(memory), get_length(memory));
        //printf("Requested memory of size %ld\n", sz);
        void* ptr = allocate_memory(nb_elements, size, memory);

        pthread_mutex_unlock(&memory_lock);
        return ptr;
    #else
        void* ptr = malloc(size*nb_elements);
        return ptr;
    #endif
}

#ifdef __CUDACC__
extern "C"
#endif
void gree(void* ptr) {
    #if defined(__CUDACC__) || defined(TEST_MEMORY_MANAGEMENT)
        pthread_mutex_lock(&memory_lock);
        memory = free_memory(ptr, memory);
        pthread_mutex_unlock(&memory_lock);
    #else
        free(ptr);
    #endif
}