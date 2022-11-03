/* This file is a copy of src/cnn/convolution.cu */
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "include/struct.h"

#define BLOCKSIZE_x 16
#define BLOCKSIZE_y 8
#define BLOCKSIZE_z 8


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

void make_convolution_cpu(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    // c'est le kernel de input
    // input[kernel->rows][kernel_k_size + output_dim-1][kernel_k_size + output_dim-1]
    // output[kernel->columns][output_dim][output_dim]
    float f;
    
    for (int i=0; i < kernel->columns; i++) {
        for (int j=0; j < output_dim; j++) {
            for (int k=0; k < output_dim; k++) {
                f = kernel->bias[i][j][k];
                for (int a=0; a < kernel->rows; a++) {
                    for (int b=0; b < kernel->k_size; b++) {
                        for (int c=0; c < kernel->k_size; c++) {
                            f += kernel->w[a][i][b][c]*input[a][j+b][k+c];
                        }
                    }
                }
                output[i][j][k] = f;
            }
        }
    }
}

#ifdef __CUDACC__
int i_div_up(int a, int b) { // Partie entière supérieure de a/b
    return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

__global__ void make_convolution_kernel(int k_size, int columns, int rows, float*** bias, size_t pitch_bias, float**** w, size_t pitch_w, float*** input, size_t pitch_input, float*** output, size_t pitch_output, int output_dim) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < kernel->columns
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < min(output_dim, k_size)
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < min(output_dim, k_size)

    int input_dim = output_dim+k_size - 1;

    if (idx >= columns || idy >= output_dim || idz >= output_dim) {
        return;
    }

    float* bias_offset;
    float* w_offset;
    float* input_offset;
    float* output_offset;

    bias_offset = (float*)((char*)bias + (idx*output_dim+idy)*pitch_bias);
    float f = bias_offset[idz];
    
    for (int a=0; a < rows; a++) {
        for (int b=0; b < k_size; b++) {
            for (int c=0; c < k_size; c++) {
                w_offset = (float*)((char*)w + ((a*columns + idx)*k_size+b)*pitch_w);
                input_offset = (float*)((char*)input + (a*input_dim + idy+b)*pitch_input);
                f += w_offset[c]*input_offset[idz+c];
            }
        }
    }

    output_offset = (float*)((char*)output + (idx*output_dim+idy)*pitch_output);
    output_offset[idz] = f;
}

void make_convolution_device(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    // Copy arrays
    size_t pitch_input;
    size_t pitch_output;
    size_t pitch_bias;
    size_t pitch_weight;
    float*** input_dev;
    float*** output_dev;
    float*** kernel_bias;
    float**** kernel_weight;

    int input_dim = output_dim+kernel->k_size - 1;
    
    // Copy ***input
    gpuErrchk( cudaMallocPitch((void**)&input_dev, &pitch_input, input_dim*sizeof(float), kernel->rows*input_dim));
    for (int i=0; i < kernel->rows; i++) {
        for (int j=0; j < input_dim; j++) {
            gpuErrchk( cudaMemcpy((void*)((char*)input_dev + (i*input_dim+j)*pitch_input), (const void*)&(input[i][j][0]), input_dim*sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    // cudaMalloc ***output
    gpuErrchk( cudaMallocPitch((void**)&output_dev, &pitch_output, output_dim*sizeof(float), kernel->columns*output_dim));

    // Copy ***Kernel bias
    gpuErrchk( cudaMallocPitch((void**)&kernel_bias, &pitch_bias, output_dim*sizeof(float), kernel->columns*output_dim));
    for (int i=0; i < kernel->columns; i++) {
        for (int j=0; j < output_dim; j++) {
            gpuErrchk( cudaMemcpy((void*)((char*)kernel_bias + (i*output_dim+j)*pitch_bias), (const void*)&(kernel->bias[i][j][0]), output_dim*sizeof(float), cudaMemcpyHostToDevice));
        }
    }

    // Copy ****Kernel weights
    gpuErrchk( cudaMallocPitch((void**)&kernel_weight, &pitch_weight, kernel->k_size*sizeof(float), (kernel->rows*kernel->columns*kernel->k_size)));
    for (int i=0; i < kernel->rows; i++) {
        for (int j=0; j < kernel->columns; j++) {
            for (int k=0; k < kernel->k_size; k++) {
                gpuErrchk( cudaMemcpy((void*)((char*)kernel_weight + ((i*kernel->columns+j)*kernel->k_size+k)*pitch_weight), (const void*)&(kernel->w[i][j][k][0]), kernel->k_size*sizeof(float), cudaMemcpyHostToDevice));
            }
        }
    }

    // Make computation
    dim3 gridSize(i_div_up(kernel->columns, BLOCKSIZE_x), i_div_up(output_dim, BLOCKSIZE_y), i_div_up(output_dim, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    make_convolution_kernel<<<gridSize, blockSize>>>(kernel->k_size, kernel->columns, kernel->rows, kernel_bias, pitch_bias, kernel_weight, pitch_weight, input_dev, pitch_input, output_dev, pitch_output, output_dim);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Copy output back
    for (int i=0; i < kernel->columns; i++) {
        for (int j=0; j < output_dim; j++) {
            gpuErrchk( cudaMemcpy((void*)&(output[i][j][0]), (const void*)((char*)output_dev + (i*output_dim+j)*pitch_output), output_dim*sizeof(float), cudaMemcpyDeviceToHost));
        }
    }

    // Free all the allocated memory
    gpuErrchk( cudaFree(input_dev) );
    gpuErrchk( cudaFree(output_dev) );
    gpuErrchk( cudaFree(kernel_bias) );
    gpuErrchk( cudaFree(kernel_weight) );
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif


void make_convolution(Kernel_cnn* kernel, float*** input, float*** output, int output_dim) {
    #ifndef __CUDACC__
    make_convolution_cpu(kernel, input, output, output_dim);
    #else
    make_convolution_device(kernel, input, output, output_dim);
    #endif
}