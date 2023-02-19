#include <stdio.h>
#include <stdbool.h>

#ifndef DEF_MEM_MANAGEMENT_H
#define DEF_MEM_MANAGEMENT_H

// A block of memory is 48kB
// https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556
#define MEMORY_BLOCK 49152


// We define our memory with a linked list of memory blocks
typedef struct Memory {
   void* start; // Start of the allocated memory
   void* cursor; // Current cursor
   size_t size; // Taille de la mémoire allouée
   int nb_alloc; // Nombre d'allocations dans le bloc
   struct Memory* next; // Élément suivant
} Memory;


/* Get memory stats for tests */
#ifdef __CUDACC__
extern "C"
#endif
/*
* Renvoie le nombre d'allocations totales dans la mémoire
*/
int get_memory_distinct_allocations();

/*
* Fonction récursive correspondante
*/
int get_distinct_allocations(Memory* mem);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Renvoie le nombre d'éléments dans la liste chaînée représentant la mémoire
*/
int get_memory_blocks_number();

/*
* Renvoie la taille d'une liste chaînée
*/
int get_length(Memory* mem);



/*
* Créer un bloc de mémoire de taille size
*/
Memory* create_memory_block(size_t size);

/*
* Allouer un élément de taille size dans mem
*/
void* allocate_memory(size_t size, Memory* mem);

/*
* Essayer de libérer le pointeur représenté par ptr dans mem
*/
Memory* free_memory(void* ptr, Memory* mem);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Alloue de la mémoire partagée CUDA si CUDA est activé
*/
void* nalloc(size_t sz);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Libérer le mémoire allouée avec nalloc
*/
void gree(void* ptr);

#endif