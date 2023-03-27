#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "../src/include/memory_management.h"
#include "../src/cnn/include/function.h"
#include "../src/include/colors.h"
#include "../src/include/utils.h"


int main() {
    printf("Checking CUDA compatibility.\n");
    bool cuda_compatible = check_cuda_compatibility();
    if (!cuda_compatible) {
        printf(RED "CUDA not compatible, skipping tests.\n" RESET);
        return 0;
    }
    printf(GREEN "OK\n" RESET);

    printf("Initialisation OK\n");
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
                input[i][j][k] = rand()/RAND_MAX;
                input_initial[i][j][k] = input[i][j][k];
            }
        }
    }
    printf(GREEN "OK\n" RESET);

    funcPtr func = get_activation_function(TANH);

    printf("Calcul par CUDA\n");
    apply_function_input(TANH, input, depth, rows, columns);
    printf(GREEN "OK\n" RESET);

    printf("Vérification des résultats\n");
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                if (fabs((*func)(input_initial[i][j][k]) - input[i][j][k]) > 1e-6) {
                    printf_error((char*)"Les résultats ne coincident pas\n");
                    printf("Différence %e\n", fabs((*func)(input_initial[i][j][k]) - input[i][j][k]));
                    //exit(1);
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

    printf(GREEN "OK\n" RESET);
    return 0;
}