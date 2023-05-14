#include <stdio.h>
#include <stdbool.h>

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


#ifdef __CUDACC__
__host__ __device__
#endif
/*
* On renvoie true si et seulement si _ et _:
* lower_bound <= x < upper_bound
* lower_bound <= y < upper_bound
*/
int not_outside(int x, int y, int lower_bound, int upper_bound);

/*
* Partie entière supérieure de a/b
*/
int i_div_up(int a, int b);

#ifdef __CUDACC__
extern "C"
#endif
/*
* Vérification de la compatibilité CUDA
*/
bool check_cuda_compatibility();

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