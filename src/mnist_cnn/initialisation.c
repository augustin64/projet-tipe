#include <stdlib.h>
#include <math.h>
#include "initialisation.h"


void initialisation_1d_matrix(int initialisation, float* matrix, int rows, int n) { //NOT FINISHED
    int i;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<rows; i++) {
        matrix[i] = lower_bound + RAND_FLT()*distance;
    }
}

void initialisation_2d_matrix(int initialisation, float** matrix, int rows, int columns, int n) { //NOT FINISHED
    int i, j;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            matrix[i][j] = lower_bound + RAND_FLT()*distance;
        }
    }
}

void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int rows, int columns, int n) { //NOT FINISHED
    int i, j, k;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<depth; i++) {
        for (j=0; j<rows; j++) {
            for (k=0; k<columns; k++) {
                matrix[i][j][k] = lower_bound + RAND_FLT()*distance;
            }
        }
    }
}

void initialisation_4d_matrix(int initialisation, float**** matrix, int rows, int columns, int rows1, int columns1, int n) { //NOT FINISHED
    int i, j, k, l;
    float lower_bound = -6/sqrt((double)n);
    float distance = -lower_bound-lower_bound;
    for (i=0; i<rows; i++) {
        for (j=0; j<columns; j++) {
            for (k=0; k<rows1; k++) {
                for (l=0; l<columns1; l++) {
                    matrix[i][j][k][l] = lower_bound + RAND_FLT()*distance;
                }
            }
        }
    }
}