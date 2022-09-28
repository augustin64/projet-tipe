#include <stdlib.h>
#include <math.h>

#include "../colors.h"
#include "include/initialisation.h"


void initialisation_1d_matrix(int initialisation, float* matrix, int rows, int n) { // TODO
    printf_warning("Appel de initialisation_1d_matrix, incomplet\n");
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (int i=0; i < rows; i++) {
        matrix[i] = lower_bound + RAND_FLT()*distance;
    }
}

void initialisation_2d_matrix(int initialisation, float** matrix, int rows, int columns, int n) { // TODO
    printf_warning("Appel de initialisation_2d_matrix, incomplet\n");
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (int i=0; i < rows; i++) {
        for (int j=0; j < columns; j++) {
            matrix[i][j] = lower_bound + RAND_FLT()*distance;
        }
    }
}

void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int rows, int columns, int n) { // TODO
    printf_warning("Appel de initialisation_3d_matrix, incomplet\n");
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (int i=0; i < depth; i++) {
        for (int j=0; j < rows; j++) {
            for (int k=0; k < columns; k++) {
                matrix[i][j][k] = lower_bound + RAND_FLT()*distance;
            }
        }
    }
}

void initialisation_4d_matrix(int initialisation, float**** matrix, int rows, int columns, int rows1, int columns1, int n) { // TODO
    printf_warning("Appel de initialisation_4d_matrix, incomplet\n");
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (int i=0; i < rows; i++) {
        for (int j=0; j < columns; j++) {
            for (int k=0; k < rows1; k++) {
                for (int l=0; l < columns1; l++) {
                    matrix[i][j][k][l] = lower_bound + RAND_FLT()*distance;
                }
            }
        }
    }
}