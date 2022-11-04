#include <stdlib.h>
#include <math.h>

#include "../include/colors.h"
#include "include/initialisation.h"

// glorot (wavier initialisation) linear, tanh, softmax, logistic (1/(fan_in+fan_out/2))
// he initialisation : RELU (2/fan_in)
// LeCun initialisation: SELU (1/fan_in)

// Only uniform for the moment
void initialisation_1d_matrix(int initialisation, float* matrix, int dim, int n_in, int n_out) {
    int n;
    if (initialisation == GLOROT) {
        n = (n_in + n_out)/2;
    
    } else if (initialisation == HE) {
        n = n_in/2; 
    } else {
        printf_warning("Initialisation non reconnue dans 'initialisation_1d_matrix' \n");
        return ;
    }
    float lower_bound = -1/sqrt((double)n);
    float distance_bounds = -2*lower_bound;
    for (int i=0; i < dim; i++) {
        matrix[i] = lower_bound + RAND_FLT()*distance_bounds;
    }
}

void initialisation_2d_matrix(int initialisation, float** matrix, int dim1, int dim2, int n_in, int n_out) {
    int n;
    if (initialisation == GLOROT) {
        n = (n_in + n_out)/2;
    
    } else if (initialisation == HE) {
        n = n_in/2; 
    } else {
        printf_warning("Initialisation non reconnue dans 'initialisation_2d_matrix' \n");
        return ;
    }
    float lower_bound = -1/sqrt((double)n);
    float distance_bounds = -2*lower_bound;
    for (int i=0; i < dim1; i++) {
        for (int j=0; j < dim2; j++) {
            matrix[i][j] = lower_bound + RAND_FLT()*distance_bounds;
        }
    }
}

void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int dim1, int dim2, int n_in, int n_out) {
    int n;
    if (initialisation == GLOROT) {
        n = (n_in + n_out)/2;
    
    } else if (initialisation == HE) {
        n = n_in/2; 
    } else {
        printf_warning("Initialisation non reconnue dans 'initialisation_3d_matrix' \n");
        return ;
    }
    float lower_bound = -1/sqrt((double)n);
    float distance_bounds = -2*lower_bound;
    for (int i=0; i < depth; i++) {
        for (int j=0; j < dim1; j++) {
            for (int k=0; k < dim2; k++) {
                matrix[i][j][k] = lower_bound + RAND_FLT()*distance_bounds;
            }
        }
    }
}

void initialisation_4d_matrix(int initialisation, float**** matrix, int depth1, int depth2, int dim1, int dim2, int n_in, int n_out) {
    int n;
    if (initialisation == GLOROT) {
        n = (n_in + n_out)/2;
    
    } else if (initialisation == HE) {
        n = n_in/2; 
    } else {
        printf_warning("Initialisation non reconnue dans 'initialisation_3d_matrix' \n");
        return ;
    }
    float lower_bound = -1/sqrt((double)n);
    float distance_bounds = -2*lower_bound;
    for (int i=0; i < depth1; i++) {
        for (int j=0; j < depth2; j++) {
            for (int k=0; k < dim1; k++) {
                for (int l=0; l < dim2; l++) {
                    matrix[i][j][k][l] = lower_bound + RAND_FLT()*distance_bounds;
                }
            }
        }
    }
}