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

/*
* Partie entière supérieure de a/b
*/
int i_div_up(int a, int b);

/*
* Vérification de la compatibilité CUDA
*/
bool check_cuda_compatibility();

#endif