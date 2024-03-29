#include <stdio.h>
#include <float.h>
#include <math.h>

#include "include/backpropagation.h"
#include "../common/include/colors.h"
#include "../common/include/utils.h"
#include "include/struct.h"

#include "include/config.h"


/*
* Softmax backward MSE
*/
#ifdef __CUDACC__
__global__ void softmax_backward_mse_kernel(float* input, float* output, int size) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= size) {
        return;
    }

    int input_val = input[idx];
    int output_val = output[idx];

    input[idx] = (output_val-input_val)*input_val*(1-input_val);
}

void softmax_backward_mse_device(float* input, float* output, int size) {
    // Make computation
    dim3 gridSize(i_div_up(size, BLOCKSIZE_x));
    dim3 blockSize(BLOCKSIZE_x);

    softmax_backward_mse_kernel<<<gridSize, blockSize>>>(input, output, size);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void softmax_backward_mse_cpu(float* input, float* output, int size) {
    /* Input et output ont la même taille */

    for (int i=0; i < size; i++){
        input[i] = (output[i]-input[i])*input[i]*(1-input[i]);
    }
}

void softmax_backward_mse(float* input, float* output, int size) {
    #ifdef __CUDACC__
    softmax_backward_mse_device(input, output, size);
    #else
    softmax_backward_mse_cpu(input, output, size);
    #endif
}


/*
* Softmax backward Cross entropy
*/
#ifdef __CUDACC__
__global__ void softmax_backward_cross_entropy_kernel(float* input, float* output, int size) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;

    if (idx >= size) {
        return;
    }

    input[idx] = output[idx] - input[idx];
}

void softmax_backward_cross_entropy_device(float* input, float* output, int size) {
    // Make computation
    dim3 gridSize(i_div_up(size, BLOCKSIZE_x));
    dim3 blockSize(BLOCKSIZE_x);

    softmax_backward_cross_entropy_kernel<<<gridSize, blockSize>>>(input, output, size);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void softmax_backward_cross_entropy_cpu(float* input, float* output, int size) {
    /* Input et output ont la même taille */

    for (int i=0; i < size; i++){
        input[i] = output[i] - input[i];
    }
}

void softmax_backward_cross_entropy(float* input, float* output, int size) {
    #ifdef __CUDACC__
    softmax_backward_cross_entropy_device(input, output, size);
    #else
    softmax_backward_cross_entropy_cpu(input, output, size);
    #endif
}


/*
* Backward average pooling
*/
#ifdef __CUDACC__
__global__ void backward_average_pooling_kernel(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < output_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < output_width

    if (idx >= depth || idy >= output_width || idz >= output_width) {
        return;
    }
    int max_move = kernel_size - padding;

    for (int a=-padding; a < max_move; a++) {
        for (int b=-padding; b < max_move; b++) {
            int idy_2 = stride*idy +a;
            int idz_2 = stride*idz +b;
            if (NOT_OUTSIDE(idy_2, idz_2, 0, input_width)) {
                int y = min(idy_2+1, min(kernel_size, input_width - idy_2));
                int z = min(idz_2+1, min(kernel_size, input_width - idz_2));
                input[idx][idy_2][idz_2] += output[idx][idy][idz]/(y*z);
            }
        }
    }
}


void backward_average_pooling_device(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    // Make computation
    dim3 gridSize(i_div_up(depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    reset_3d_array(input, depth, input_width, input_width);

    backward_average_pooling_kernel<<<gridSize, blockSize>>>(input, output, input_width, output_width, depth, kernel_size, stride, padding);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void backward_average_pooling_cpu(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    /* Input et output ont la même profondeur (depth) */

    reset_3d_array(input, depth, input_width, input_width);
    int max_move = kernel_size - padding;

    for (int i=0; i < depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                for (int a=-padding; a < max_move; a++) {
                    for (int b=-padding; b < max_move; b++) {
                        int j_2 = stride*j +a;
                        int k_2 = stride*k + b;
                        if (NOT_OUTSIDE(j_2, k_2, 0, input_width)){
                            int j_3 = min(j_2+1, min(kernel_size, input_width - j_2));
                            int k_3 = min(k_2+1, min(kernel_size, input_width - k_2));
                            input[i][j_2][k_2] += output[i][j][k]/(j_3*k_3);
                        }
                    }
                }
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void backward_average_pooling(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    #ifndef __CUDACC__
    backward_average_pooling_cpu(input, output, input_width, output_width, depth, kernel_size, stride, padding);
    #else
    backward_average_pooling_device(input, output, input_width, output_width, depth, kernel_size, stride, padding);
    #endif
}


/*
* Backward max pooling
*/
#ifdef __CUDACC__
__global__ void backward_max_pooling_kernel(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < output_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < output_width

    if (idx >= depth || idy >= output_width || idz >= output_width) {
        return;
    }
    int max_move = kernel_size - padding;
    float m = -FLT_MAX;
    int a_max = -1;
    int b_max = -1;
    int cpt = 0;

    for (int a=-padding; a < max_move; a++) {
        for (int b=-padding; b < max_move; b++) {
            int idy_2 = stride*idy +a;
            int idz_2 = stride*idz +b;
            if (NOT_OUTSIDE(idy_2, idz_2, 0, input_width)) {
                if (input[idx][idy_2][idz_2] > m) {
                    m = input[idx][idy_2][idz_2];
                    a_max = a;
                    b_max = b;
                }
                input[idx][idy_2][idz_2] = 0;
                cpt++;
            }
        }
    }
    if (cpt==0) {
        printf(RED "[ERROR]" RESET " Dimensions ou stride ou padding erroné dans 'backward_max_pooling_cpu'\n");
    }
    input[idx][stride*idy +a_max][stride*idz +b_max] = output[idx][idy][idz]/cpt;
}


void backward_max_pooling_device(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    // Make computation
    dim3 gridSize(i_div_up(depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_z));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    backward_max_pooling_kernel<<<gridSize, blockSize>>>(input, output, input_width, output_width, depth, kernel_size, stride, padding);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void backward_max_pooling_cpu(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    float m; // Maximum
    int a_max, b_max; // Indices du maximum
    int cpt;
    int max_move = kernel_size - padding;

    for (int i=0; i < depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                m = -FLT_MAX;
                a_max = -1;
                b_max = -1;
                cpt = 0;

                for (int a=-padding; a < max_move; a++) {
                    for (int b=-padding; b < max_move; b++) {
                        int j_2 = stride*j +a;
                        int k_2 = stride*k +b;
                        if (NOT_OUTSIDE(j_2, k_2, 0, input_width)) {
                            if (input[i][j_2][k_2] > m) {
                                m = input[i][j_2][k_2];
                                a_max = a;
                                b_max = b;
                            }
                            input[i][j_2][k_2] = 0;
                            cpt++;
                        }
                    }
                }
                if (cpt==0) {
                    printf_error((char*)"Dimensions ou stride ou padding erroné dans 'backward_max_pooling_cpu'\n");
                }
                else {
                    input[i][stride*j +a_max][stride*k +b_max] = output[i][j][k]/cpt;
                }
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void backward_max_pooling(float*** input, float*** output, int input_width, int output_width, int depth, int kernel_size, int stride, int padding) {
    #ifndef __CUDACC__
    backward_max_pooling_cpu(input, output, input_width, output_width, depth, kernel_size, stride, padding);
    #else
    backward_max_pooling_device(input, output, input_width, output_width, kernel_size, depth, stride, padding);
    #endif
}

/*
* Backward Dense
*/
#ifdef __CUDACC__
__global__ void backward_dense_kernel_1(float** d_weights, float* d_bias, float* input, float* output, int size_input, int size_output) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < size_input
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < size_output

    if (idx >= size_input || idy >= size_output) {
        return;
    }

    if (idx == 0) {
        d_bias[idy] += output[idy];
    }
    d_weights[idx][idy] += input[idx]*output[idy];
}

__global__ void backward_dense_kernel_2(float** weights, float* input, float* input_z, float* output, int size_input, int size_output, funcPtr d_f) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < size_input

    if (idx >= size_input) {
        return;
    }

    float tmp=0;
    for (int j=0; j < size_output; j++) {
        tmp += output[j]*weights[idx][j];
    }
    input[idx] = tmp*( (*d_f)(input_z[idx]) );
}

void backward_dense_device(Kernel_nn* ker, D_Kernel_nn* d_ker, float* input, float* input_z, float* output, int size_input, int size_output, int activation, int is_first) {
    // Make computation
    dim3 gridSize1(i_div_up(size_input, BLOCKSIZE_x), i_div_up(size_output, BLOCKSIZE_y));
    dim3 blockSize1(BLOCKSIZE_x, BLOCKSIZE_y);

    backward_dense_kernel_1<<<gridSize1, blockSize1>>>(d_ker->d_weights, d_ker->d_bias, input, output, size_input, size_output);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Second kernel
    if (is_first != 1) {
        dim3 gridSize1(i_div_up(size_input, BLOCKSIZE_x));
        dim3 blockSize1(BLOCKSIZE_x);

        funcPtr d_function = get_activation_function_cuda(activation);

        backward_dense_kernel_2<<<gridSize1, blockSize1>>>(ker->weights, input, input_z, output, size_input, size_output, d_function);
        
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
}
#endif

void backward_dense_cpu(Kernel_nn* ker, D_Kernel_nn* d_ker, float* input, float* input_z, float* output, int size_input, int size_output, int activation, int is_first) {

    funcPtr d_function = get_activation_function(activation);
    // Bias
    for (int j=0; j < size_output; j++) {
        d_ker->d_bias[j] += output[j];
    }

    // Weights
    for (int i=0; i < size_input; i++) {
        for (int j=0; j < size_output; j++) {
            d_ker->d_weights[i][j] += input[i]*output[j];
        }
    }

    // Input
    if (is_first==1) {// Pas besoin de backpropager dans l'input
        return;
    }

    for (int i=0; i < size_input; i++) {
        float tmp=0;
        for (int j=0; j < size_output; j++) {
            tmp += output[j]*ker->weights[i][j];
        }
        input[i] = tmp*d_function(input_z[i]);
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void backward_dense(Kernel_nn* ker, D_Kernel_nn* d_ker, float* input, float* input_z, float* output, int size_input, int size_output, int activation, int is_first) {
    #ifndef __CUDACC__
    backward_dense_cpu(ker, d_ker, input, input_z, output, size_input, size_output, activation, is_first);
    #else
    backward_dense_device(ker, d_ker, input, input_z, output, size_input, size_output, activation, is_first);
    #endif
}



/*
* Backward linearisation
*/
#ifdef __CUDACC__
__global__ void backward_linearisation_kernel_1(float** d_weights, float* d_bias, float*** input, float* output, int input_depth, int input_width, int size_output) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < input_depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < input_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < input_width

    if (idx >= input_depth || idy >= input_width || idz >= input_width) {
        return;
    }

    int id = idx*input_width*input_width + idy*input_width + idz;
    
    for (int j=0; j < size_output; j++) {
        d_weights[id][j] += input[idx][idy][idz]*output[j];
    }
    if (id == 0) {
        for (int j=0; j < size_output;  j++) {
            d_bias[j] += output[j];
        }
    }
}

__global__ void backward_linearisation_kernel_2(float** weights, float*** input, float*** input_z, float* output, int input_depth, int input_width, int size_output, funcPtr d_f) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < input_depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < input_width
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < input_width

    if (idx >= input_depth || idy >= input_width || idz >= input_width) {
        return;
    }
    int id = (idx*input_width+idy)*input_width + idz;

    float tmp=0;
    for (int j=0; j < size_output; j++) {
        tmp += output[j]*weights[id][j];
    }
    input[idx][idy][idz] = tmp*( (*d_f)(input_z[idx][idy][idz]) );
}

void backward_linearisation_device(Kernel_nn* ker, D_Kernel_nn* d_ker, float*** input, float*** input_z, float* output, int input_depth, int input_width, int size_output, int activation) {
    // Make computation
    dim3 gridSize(i_div_up(input_depth, BLOCKSIZE_x), i_div_up(input_width, BLOCKSIZE_y), i_div_up(input_width, BLOCKSIZE_y));
    dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    backward_linearisation_kernel_1<<<gridSize, blockSize>>>(d_ker->d_weights, d_ker->d_bias, input, output, input_depth, input_width, size_output);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // Second kernel
    funcPtr d_function = get_activation_function_cuda(activation);

    backward_linearisation_kernel_2<<<gridSize, blockSize>>>(ker->weights, input, input_z, output, input_depth, input_width, size_output, d_function);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );
}
#endif

void backward_linearisation_cpu(Kernel_nn* ker, D_Kernel_nn* d_ker, float*** input, float*** input_z, float* output, int input_depth, int input_width, int size_output, int activation) {
   
    funcPtr d_function = get_activation_function(activation);

    // Bias
    for (int j=0; j < size_output; j++) {
        d_ker->d_bias[j] += output[j];
    }

    // Weights
    int cpt = 0;
    for (int i=0; i < input_depth; i++) {
        for (int k=0; k < input_width; k++) {
            for (int l=0; l < input_width; l++) {
                for (int j=0; j < size_output; j++) {
                    d_ker->d_weights[cpt][j] += input[i][k][l]*output[j];
                }
                cpt++;
            }
        }
    }

    // Input
    cpt = 0;
    for (int i=0; i < input_depth; i++) {
        for (int k=0; k < input_width; k++) {
            for (int l=0; l < input_width; l++) {
                float tmp=0;
                for (int j=0; j < size_output; j++) {
                    tmp += output[j]*ker->weights[cpt][j];
                }
                input[i][k][l] = tmp*d_function(input_z[i][k][l]);
                cpt++;
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void backward_linearisation(Kernel_nn* ker, D_Kernel_nn* d_ker, float*** input, float*** input_z, float* output, int input_depth, int input_width, int size_output, int activation) {
    #ifndef __CUDACC__
    backward_linearisation_cpu(ker, d_ker, input, input_z, output, input_depth, input_width, size_output, activation);
    #else
    backward_linearisation_device(ker, d_ker, input, input_z, output, input_depth, input_width, size_output, activation);
    #endif
}

/*
* Backward convolution
*/
#ifdef __CUDACC__
__global__ void backward_convolution_dbias_kernel(float*** d_bias, float*** output, int output_depth, int output_width) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;
    int idz = threadIdx.z + blockDim.z*blockIdx.z;
    
    if (idx >= output_depth || idy >= output_width || idz >= output_width) {
        return;
    }
    d_bias[idx][idy][idz] += output[idx][idy][idz];
}

__global__ void backward_convolution_dweight_kernel(float**** d_weights, float*** input, float*** output, int input_depth, int output_depth, int input_width, int output_width, int k_size, int stride, int padding) {
    /*
    * L'ordre des boucles a été changé par rapport à l'implémentation sur CPU
    * afin d'utiliser possiblement plus de coeurs à la fois (car en général, depth << width)
    * En gardant les indices des boucles sur CPU notées h,i,j,k,l,m; on fait donc l,m,i,h,j,k 
    */
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // l
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // m
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // i

    if (idx >= output_width || idy >= output_width || idz >= output_depth) {
        return;
    }

    int max_move = k_size - padding;
    for (int h=0; h < input_depth; h++) {
        for (int j=-padding; j < max_move; j++) {
            for (int k=-padding; k < max_move; k++) {
                if (NOT_OUTSIDE(idx*stride+j, idy*stride+k, 0, input_width)) {
                    atomicAdd(&d_weights[h][idz][j+padding][k+padding], input[h][idx*stride+j][idy*stride+k]*output[idz][idx][idy]);
                }
            }
        }
    }
}

__global__ void backward_convolution_propagate_kernel(float**** weights, float*** input, float*** output, int input_depth, int input_width, int output_width, int output_depth, int k_size, int stride, int padding) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;

    if (idx >= input_depth || idy >= output_depth) {
        return;
    }
    int max_move = k_size - padding;
    for (int j=-padding; j < max_move; j++) {
        for (int k=-padding; k < max_move; k++) {
            for (int l=0; l < output_width; l++) {
                for (int m=0; m < output_width; m++) {
                    if (NOT_OUTSIDE(l*stride+j, m*stride+k, 0, input_width)) {
                        atomicAdd(&input[idx][l*stride+j][m*stride+k], output[idy][l][m]*weights[idx][idy][j+padding][k+padding]);
                    }
                }
            }
        }
    }
}

__global__ void backward_convolution_apply_propagate_kernel(float*** input, float*** input_z, int input_depth, int input_width, funcPtr d_f) {
    int idx = threadIdx.x + blockDim.x*blockIdx.x;
    int idy = threadIdx.y + blockDim.y*blockIdx.y;

    if (idx >= input_depth || idy >= input_width) {
        return;
    }

    for (int k=0; k < input_width; k++) {
        input[idx][idy][k] = input[idx][idy][k]*d_f(input_z[idx][idy][k]);
    }
}

void backward_convolution_device(Kernel_cnn* kernel, D_Kernel_cnn* d_kernel, float*** input, float*** input_z, float*** output, int input_depth, int input_width, int output_depth, int output_width, int activation, int is_first, int kernel_size, int padding, int stride) {
    // Bias Kernel
    dim3 gridSize1(i_div_up(output_depth, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_width, BLOCKSIZE_y));
    dim3 blockSize1(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    backward_convolution_dbias_kernel<<<gridSize1, blockSize1>>>(d_kernel->d_bias, output, output_depth, output_width);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    dim3 gridSize2(i_div_up(output_width, BLOCKSIZE_x), i_div_up(output_width, BLOCKSIZE_y), i_div_up(output_depth, BLOCKSIZE_y));
    dim3 blockSize2(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

    backward_convolution_dweight_kernel<<<gridSize2, blockSize2>>>(d_kernel->d_weights, input, output, input_depth, output_depth, input_width, output_width, kernel_size, stride, padding);
    
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // input propagation Kernel
    if (is_first != 1) {
        reset_3d_array(input, input_depth, input_width, input_width);

        dim3 gridSize3(i_div_up(input_depth, BLOCKSIZE_x), i_div_up(output_depth, BLOCKSIZE_y));
        dim3 blockSize3(BLOCKSIZE_x, BLOCKSIZE_y);

        backward_convolution_propagate_kernel<<<gridSize3, blockSize3>>>(kernel->weights, input, output, input_depth, input_width, output_width, output_depth, kernel_size, stride, padding);

        dim3 gridSize4(i_div_up(input_depth, BLOCKSIZE_x), i_div_up(input_width, BLOCKSIZE_y));
        dim3 blockSize4(BLOCKSIZE_x, BLOCKSIZE_y);

        funcPtr d_function = get_activation_function_cuda(activation);

        backward_convolution_apply_propagate_kernel<<<gridSize4, blockSize4>>>(input, input_z, input_depth, input_width, d_function);

        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
}
#endif


void backward_convolution_cpu(Kernel_cnn* ker, D_Kernel_cnn* d_ker, float*** input, float*** input_z, float*** output, int input_depth, int input_width, int output_depth, int output_width, int activation, int is_first, int kernel_size, int padding, int stride) {
    
    funcPtr d_function = get_activation_function(activation);
    int max_move = kernel_size - padding;

    // Bias
    for (int i=0; i < output_depth; i++) {
        for (int j=0; j < output_width; j++) {
            for (int k=0; k < output_width; k++) {
                d_ker->d_bias[i][j][k] += output[i][j][k];
            }
        }
    }

    // Weights    
    for (int h=0; h < input_depth; h++) {
        for (int i=0; i < output_depth; i++) {
            for (int j=-padding; j < max_move; j++) {
                for (int k=-padding; k < max_move; k++) {
                    float tmp = 0;
                    for (int l=0; l < output_width; l++) {
                        for (int m=0; m < output_width; m++) {
                            if (NOT_OUTSIDE(l*stride+j, m*stride+k, 0, input_width)) {
                                tmp += input[h][l*stride+j][m*stride+k]*output[i][l][m];
                            }
                        }
                    }
                    d_ker->d_weights[h][i][j+padding][k+padding] += tmp;
                }
            }
        }
    }

    // Input
    if (is_first==1) // Pas besoin de backpropager dans l'input
        return;
    for (int i=0; i < input_depth; i++) {
        for (int j=0; j < input_width; j++) {
            for (int k=0; k < input_width; k++) {
                input[i][j][k] = 0;
            }
        }
    }
    for (int h=0; h < input_depth; h++) {
        for (int i=0; i < output_depth; i++) {
            for (int j=-padding; j < max_move; j++) {
                for (int k=-padding; k < max_move; k++) {
                    for (int l=0; l < output_width; l++) {
                        for (int m=0; m < output_width; m++) {
                            if (NOT_OUTSIDE(l*stride+j, m*stride+k, 0, input_width)) {
                                input[h][l*stride+j][m*stride+k] += output[i][l][m]*ker->weights[h][i][j+padding][k+padding];
                            }
                        }
                    }
                }
            }
        }
    }
    for (int i=0; i < input_depth; i++) {
        for (int j=0; j < input_width; j++) {
            for (int k=0; k < input_width; k++) {
                input[i][j][k] = input[i][j][k]*d_function(input_z[i][j][k]);
            }
        }
    }
}

#ifdef __CUDACC__
extern "C"
#endif
void backward_convolution(Kernel_cnn* ker, D_Kernel_cnn* d_ker, float*** input, float*** input_z, float*** output, int input_depth, int input_width, int output_depth, int output_width, int activation, int is_first, int kernel_size, int padding, int stride) {
    #ifndef __CUDACC__
    backward_convolution_cpu(ker, d_ker, input, input_z, output, input_depth, input_width, output_depth, output_width, activation, is_first, kernel_size, padding, stride);
    #else
    backward_convolution_device(ker, d_ker, input, input_z, output, input_depth, input_width, output_depth, output_width, activation, is_first, kernel_size, padding, stride);
    #endif
}