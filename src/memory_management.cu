#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>
#include <pthread.h>

#include "include/memory_management.h"
#include "include/colors.h"


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


Memory* create_memory_block(size_t size) {
    Memory* mem = (Memory*)malloc(sizeof(Memory));
    #ifdef __CUDACC__
    cudaMallocManaged(&(mem->start), size, cudaMemAttachHost);
    #else
    mem->start = malloc(size);
    #endif
    mem->cursor = mem->start;
    mem->size = size;
    mem->nb_alloc = 0;
    mem->next = NULL;
    
    return mem;
}


void* allocate_memory(size_t size, Memory* mem) {
    // Si il y a suffisamment de mémoire disponible
    if (mem->size - ((intptr_t)mem->cursor - (intptr_t)mem->start) >= size) {
        void* ptr = mem->cursor;
        mem->cursor = (void*)((intptr_t)mem->cursor + size); // On décale le curseur de la taille allouée
        mem->nb_alloc++;
        return ptr;
    } else {
        //printf("Mémoire disponible: %ld. Nécessaire: %ld\n", mem->size - ((intptr_t)mem->cursor - (intptr_t)mem->start), size);
        // Sinon on continue sur l'élément suivant de la liste
        if (!mem->next) {
            mem->next = create_memory_block(MEMORY_BLOCK < size ? size : MEMORY_BLOCK);
        }
        return allocate_memory(size, mem->next);
    }
}


Memory* free_memory(void* ptr, Memory* mem) {
    if (!mem) {
        printf("[ERREUR] Le pointeur %p a déjà été libéré ou n'a jamais été alloué\n", ptr);
        return mem;
    }
    if ((intptr_t)mem->start <= (intptr_t)ptr && (intptr_t)ptr <= (intptr_t)mem->start + (intptr_t)mem->size) {
        mem->nb_alloc--;
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
void* nalloc(size_t sz) {
    #if defined(__CUDACC__) || defined(TEST_MEMORY_MANAGEMENT)
        pthread_mutex_lock(&memory_lock);
        if (!memory) {
            // We allocate a new memory block
            memory = create_memory_block(MEMORY_BLOCK < sz ? sz : MEMORY_BLOCK);
        }
        //printf("Distinct allocations: %d Blocks: %d\n", get_distinct_allocations(memory), get_length(memory));
        //printf("Requested memory of size %ld\n", sz);
        void* ptr = allocate_memory(sz, memory);

        pthread_mutex_unlock(&memory_lock);
        return ptr;
    #else
        void* ptr = malloc(sz);
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