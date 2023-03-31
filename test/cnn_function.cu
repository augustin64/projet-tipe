#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "../src/include/memory_management.h"
#include "../src/cnn/include/function.h"
#include "../src/include/colors.h"
#include "../src/include/utils.h"

#include "../src/cnn/include/config.h"

__global__ void local_kernel(funcPtr f, float*** input, int depth, int rows, int columns) {
    // Équivalents respectifs de i, j et k dans la boucle effectuée par le cpu
    int idx = threadIdx.x + blockDim.x*blockIdx.x; // < depth
    int idy = threadIdx.y + blockDim.y*blockIdx.y; // < rows
    int idz = threadIdx.z + blockDim.z*blockIdx.z; // < columns

    if (idx >= depth || idy >= rows || idz >= columns) {
        return;
    }

    input[idx][idy][idz] = (*f)(input[idx][idy][idz]);
}


void test1(int activation, bool use_local_kernel) {
    printf("Test sur la fonction %d\n", activation);
    printf("\tInitialisation OK\n");
    // Initialise values
    int depth = 10;
    int rows = 10;
    int columns = 10;

    float*** input = (float***)nalloc(depth, sizeof(float**));
    float*** input_initial = (float***)malloc(depth*sizeof(float**));
    for (int i=0; i < depth; i++) {
        input[i] = (float**)nalloc(rows, sizeof(float*));
        input_initial[i] = (float**)malloc(rows*sizeof(float*));
        for (int j=0; j < rows; j++) {
            input[i][j] = (float*)nalloc(columns, sizeof(float));
            input_initial[i][j] = (float*)malloc(columns*sizeof(float));
            for (int k=0; k < columns; k++) {
                input[i][j][k] = rand()/(float)RAND_MAX;
                input_initial[i][j][k] = input[i][j][k];
            }
        }
    }
    printf("\t" GREEN "OK\n" RESET);

    funcPtr func_cpu = get_activation_function(activation);

    if (!use_local_kernel) {
        printf("\tCalcul par CUDA\n");
        apply_function_input(activation, input, depth, rows, columns);
    } else {
        printf("\tCalcul par CUDA sur le kernel local\n");
        dim3 gridSize(i_div_up(depth, BLOCKSIZE_x), i_div_up(rows, BLOCKSIZE_y), i_div_up(columns, BLOCKSIZE_z));
        dim3 blockSize(BLOCKSIZE_x, BLOCKSIZE_y, BLOCKSIZE_z);

        funcPtr function_cuda = get_activation_function_cuda(activation);

        local_kernel<<<gridSize, blockSize>>>(function_cuda, input, depth, rows, columns);
        
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
    }
    printf("\t" GREEN "OK\n" RESET);

    printf("\tVérification des résultats\n");
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                if (fabs((*func_cpu)(input_initial[i][j][k]) - input[i][j][k]) > 1e-6) {
                    printf_error((char*)"Les résultats ne coincident pas\n");
                    printf("Différence %e\n", fabs((*func_cpu)(input_initial[i][j][k]) - input[i][j][k]));
                    exit(1);
                }
            }
            gree(input[i][j]);
            free(input_initial[i][j]);
        }
        gree(input[i]);
        free(input_initial[i]);
    }
    gree(input);
    free(input_initial);

    printf("\t" GREEN "OK\n" RESET);
    printf(GREEN "OK\n" RESET);
}

int main() {
    printf("Checking CUDA compatibility.\n");
    bool cuda_compatible = check_cuda_compatibility();
    if (!cuda_compatible) {
        printf(RED "CUDA not compatible, skipping tests.\n" RESET);
        return 0;
    }
    printf(GREEN "OK\n" RESET);

    for (int i=1; i < 7; i++) {
        if (i != 5) { // Exclude SOFTMAX
            test1(i, false); // use function i
            test1(-i, false); // use function i'
            test1(i, true); // use function i in the kernel declared in this file
            test1(-i, true); // use function i' in the kernel declared in this file
        }
    }
    return 0;
}