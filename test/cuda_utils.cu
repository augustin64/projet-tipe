#include <stdlib.h>
#include <stdio.h>

#ifdef __CUDACC__
#   warning compiling for CUDA
#   include "../src/mnist/cuda_utils.cu"
#   define MAX_CUDA_THREADS 1024 // from NVIDIA documentation
#else
#   warning skipping CUDA checks
#endif

#ifdef __CUDACC__
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void check_labels(int n, unsigned int* labels) {
    for (int i=0; i < n; i++) {
        (void)labels[i];
    }
}


int main() {
    printf("Test de la compatibilité CUDA\n");
    check_cuda_compatibility();
    printf("OK\n");

    printf("Lecture des labels\n");
    unsigned int* labels = cudaReadMnistLabels("data/mnist/t10k-labels-idx1-ubyte");
    printf("OK\n");

    printf("Test des labels\n");
    //! TODO: fix
    // Ne provoque pas d'erreurs, mais tous les labels valent 1
    check_labels<<<1, 1>>>(10000, labels);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
    printf("OK\n");
    
    return 0;
}
#else
int main() {
    printf("Pas de test CUDA à passer\n");
    return 0;
}
#endif