#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <time.h>

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 16

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

float random_float(float low, float high) {
  float t = (float)rand() / (float)RAND_MAX;
  return (1.0f - t) * low + t * high;
}


void fill_matrix_random(float **matrix, int n, int p) {
  for (int i=0; i < n; i++) {
    for (int j=0; j < p; j++) {
        matrix[i][j] = random_float(0.0f, 15.0f);
    }
  }
}


void print_matrix(float** mat, int n, int p) {
    for (int i=0; i < n; i++) {
        printf("[\t");
        for (int j=0; j < p; j++) {
            printf("%0.1f\t", mat[i][j]);
        }
        printf("]\n");
    }
}


float** create_matrix(int n, int p) {
    float** matrix = (float**)malloc(n*sizeof(float*));
    for (int i=0; i < n; i++) {
        matrix[i] = (float*)malloc(sizeof(float)*p);
    }

    fill_matrix_random(matrix, n, p);
    return matrix;
}


float** create_empty_matrix(int n, int p) {
    float** matrix = (float**)malloc(n*sizeof(float*));
    for (int i=0; i < n; i++) {
        matrix[i] = (float*)malloc(p*sizeof(float));
        for (int j=0; j < p; j++) {
            matrix[i][j] = 0.;
        }
    }
    return matrix;
}


#ifdef __CUDACC__
int i_div_up(int hostPtr, int b){
    return ((hostPtr % b) != 0) ? (hostPtr / b + 1) : (hostPtr / b);
}


__global__ void matrix_mul_kernel(float* Md, float* Nd, float* Pd, int n, int p, int q, size_t pitch_m, size_t pitch_n, size_t pitch_p) {
    // 2D Thread ID
    int tx = blockIdx.x*blockDim.x + threadIdx.x;
    int ty = blockIdx.y*blockDim.y + threadIdx.y;
    // Pvalue stores the Pd element that is computed by the thread
    float Pvalue = 0.;
    float* M_offset;
    float* N_offset;

    for (int k = 0; k < p; k++) {
        M_offset = (float *)((char*)Md + ty * pitch_m);
        N_offset = (float *)((char*)Nd + k * pitch_n);
    
        Pvalue += M_offset[k] * N_offset[tx];
    }
    // Write the matrix to device memory each thread writes one element
    float* P_offset = (float*)((char*)Pd + ty * pitch_p);
    P_offset[tx] = Pvalue;
}


void matrix_multiplication_device(float** m1, float** m2, float** result, int n, int p, int q) {
    // Préparation des matrices
    size_t pitch_m1_dev;
    size_t pitch_m2_dev;
    size_t pitch_result_dev;
    float* m1_dev;
    float* m2_dev;
    float* result_dev;
    
    gpuErrchk( cudaMallocPitch((void**)&m1_dev, &pitch_m1_dev, p * sizeof(float), n));
    for (int i=0; i < n; i++) {
        gpuErrchk( cudaMemcpy2D((void*)((char*)m1_dev + i*pitch_m1_dev), pitch_m1_dev, (const void*)&(m1[i][0]), p*sizeof(float), p*sizeof(float), 1, cudaMemcpyHostToDevice));
    }
    
    gpuErrchk( cudaMallocPitch((void**)&m2_dev, &pitch_m2_dev, q * sizeof(float), p));
    for (int i=0; i < p; i++) {
        gpuErrchk( cudaMemcpy2D((void*)((char*)m2_dev + i*pitch_m2_dev), pitch_m2_dev, (const void*)&(m2[i][0]), q*sizeof(float), q*sizeof(float), 1, cudaMemcpyHostToDevice));
    }

    gpuErrchk( cudaMallocPitch((void**)&result_dev, &pitch_result_dev, q * sizeof(float), n));

    // Traitement
    dim3 gridSize(i_div_up(n, BLOCKSIZE_x), i_div_up(q, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_y, BLOCKSIZE_x);

    matrix_mul_kernel<<<gridSize, blockSize>>>(m1_dev, m2_dev, result_dev, n, p, q, pitch_m1_dev, pitch_m2_dev, pitch_result_dev);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Post-traitement
    for (int i=0; i < n; i++) {
        gpuErrchk( cudaMemcpy2D((void*)&(result[i][0]), q*sizeof(float), (const void*)((char*)result_dev + i*pitch_result_dev), pitch_result_dev, sizeof(float)*q, 1, cudaMemcpyDeviceToHost));
    }

    gpuErrchk( cudaFree(result_dev) );
    gpuErrchk( cudaFree(m1_dev) );
    gpuErrchk( cudaFree(m2_dev) );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif


bool check_cuda_compatibility() {
    #ifdef __CUDACC__
    int nDevices;
    cudaDeviceProp prop;

    cudaGetDeviceCount(&nDevices);
    if (nDevices == 0) {
        printf("Pas d'utilisation du GPU\n\n");
        return false;
    }

    printf("GPUs disponibles:\n");

    for (int i=0; i < nDevices; i++) {
        cudaGetDeviceProperties(&prop, i);
        printf(" - %s\n", prop.name);
    }

    cudaGetDeviceProperties(&prop, 0);
    printf("Utilisation du GPU: %s\n\n", prop.name);
    return true;
    #else
    printf("Pas d'utilisation du GPU\n\n");
    return false;
    #endif
}


void matrix_multiplication_host(float** m1, float** m2, float** result, int n, int p, int q) {
    for (int i=0; i < n; i++) {
        for (int j=0; j < q; j++) {
            result[i][j] = 0.;
            for (int k=0; k < p; k++) {
                result[i][j] += m1[i][k] + m2[k][j];
            }
        }
    }
}


void matrix_multiplication(float** m1, float** m2, float** result, int n, int p, int q, bool use_cuda) {
    #ifdef __CUDACC__
    if (use_cuda) {
        matrix_multiplication_device(m1, m2, result, n, p, q);
    } else {
        matrix_multiplication_host(m1, m2, result, n, p, q);
    }
    #else
    matrix_multiplication_host(m1, m2, result, n, p, q);
    #endif
}


int main() {
    srand(time(NULL));
    int n = 3;
    int p = 3;
    int q = 3;
    float** matrix1 = create_matrix(n, p);
    float** matrix2 = create_matrix(p, q);
    float** result = create_empty_matrix(n, q);

    clock_t start, end;
    double cpu_time_used;

    start = clock();
    matrix_multiplication(matrix1, matrix2, result, n, p, q, check_cuda_compatibility());
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time used: %lf seconds\n", cpu_time_used);

    print_matrix(matrix1, n, p);
    printf("\n");
    print_matrix(matrix2, p, q);
    printf("\n");
    print_matrix(result, n, q);

    return 0;
}