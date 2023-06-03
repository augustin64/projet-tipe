#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef USE_CUDA
   #ifndef __CUDACC__
      #include "cuda_runtime.h"
   #endif
#else
    #ifdef __CUDACC__
        #define USE_CUDA
    #endif
#endif

#ifndef DEF_UTILS_CU_H
#define DEF_UTILS_CU_H

#ifdef __CUDACC__
/* CUDA memcheck */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
#endif

#define NOT_OUTSIDE(x, y, lower_bound, upper_bound) !(x < lower_bound || y < lower_bound || x >= upper_bound || y>= upper_bound)

#ifndef __CUDACC__
/*
* Renvoie la valeur minimale entre a et b
*/
int min(int a, int b);

/*
* Renvoie la valeur maximale entre a et b
*/
int max(int a, int b);
#endif


/*
* Partie entière supérieure de a/b
*/
int i_div_up(int a, int b);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Vérification de la compatibilité CUDA
* spécifier avec "verbose" si il faut afficher
* la carte utilisée notamment
*/
bool cuda_setup(bool verbose);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Copier des valeurs d'un tableau de dimension 3 de mémoire partagée
*/
void copy_3d_array(float*** source, float*** dest, int dimension1, int dimension2, int dimension3);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Remplir un tableau de 0.
*/
void reset_3d_array(float*** source, int dimension1, int dimension2, int dimension3);
#endif