#include <stdio.h>
#include <stdbool.h>

#ifndef DEF_MEM_MANAGEMENT_H
#define DEF_MEM_MANAGEMENT_H

// A block of memory is 48kB
// https://forums.developer.nvidia.com/t/find-the-limit-of-shared-memory-that-can-be-used-per-block/48556
#define MEMORY_BLOCK 49152

// On n'alloue de la mémoire que dans le dernier bloc créé, on ne parcourt donc pas la liste
// Cela augmente légèrement l'utilisation de la mémoire, mais permet un gain de temps conséquent
// Pour VGG16, environ 1% de mémoire supplémentaire utilisée,
// L'initialisation passe de 1h02 à 2.4s sur mon matériel
#define MEMORY_TAIL_OPT

// We define our memory with a linked list of memory blocks
typedef struct Memory {
   void* start; // Start of the allocated memory
   void* cursor; // Current cursor
   size_t size; // Taille de la mémoire allouée
   int nb_alloc; // Nombre d'allocations dans le bloc
   unsigned int id; // Nombre aléatoire permettant d'identifier le bloc plus facilement lors du débogage
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
* Appels récursifs de la fonction print_memory
*/
void print_memory_rec(Memory* mem);

/*
* Affiche les blocs actuels de la mémoire ainsi que leur utilisation
*/
void print_memory();


#ifdef __CUDACC__
extern "C"
#endif
/*
* Supprime tous les blocs de mémoire
*/
void free_all_memory();

/*
* Fonction récursive correspondante
*/
void free_all_memory_rec(Memory* mem);


/*
* Créer un bloc de mémoire de taille size
*/
Memory* create_memory_block(size_t size);

/*
* Allouer un élément de taille size dans mem
*/
void* allocate_memory(int nb_elements, size_t size, Memory* mem);

/*
* Essayer de libérer le pointeur représenté par ptr dans mem
* Si `already_freed`, le programme ne renvoiera pas d'erreur si
* le bloc correspondant à l'élément est déjà libéré
* (dans l'utilisation de `free_all_memory()` par exemple)
*/
Memory* free_memory(void* ptr, Memory* mem, bool already_freed);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Alloue de la mémoire partagée CUDA si CUDA est activé
*/
void* nalloc(int nb_elements, size_t size);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Libérer le mémoire allouée avec nalloc
* Si `already_freed`, le programme ne renvoiera pas d'erreur si
* le bloc correspondant à l'élément est déjà libéré
* (dans l'utilisation de `free_all_memory()` par exemple)
*/
void gree(void* ptr, bool already_freed);

#endif