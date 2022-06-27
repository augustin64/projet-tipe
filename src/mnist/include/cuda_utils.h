#include <stdio.h>
#include <stdlib.h>

#ifndef DEF_CUDA_UTILS_H
#define DEF_CUDA_UTILS_H
/*
* Il est entendu par "device" le GPU supportant CUDA utilisé
*/

/*
* Lecture des labels et écriture dans la mémoire du device
*/
unsigned int* cudaReadMnistLabels(char* label_file);

/*
* Vérification de la disponibilité d'un device
*/
void check_cuda_compatibility();

#endif