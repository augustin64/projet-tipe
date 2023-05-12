#include <stdlib.h>
#include <math.h>

#include "../common/include/colors.h"
#include "include/initialisation.h"

// glorot (wavier initialisation) linear, tanh, softmax, logistic (1/(fan_in+fan_out/2))
// he initialisation : RELU (2/fan_in)
// LeCun initialisation: SELU (1/fan_in)

// Explained in https://machinelearningmastery.com/weight-initialization-for-deep-learning-neural-networks/

float randn() {
    float f1=0.;
    while (f1 == 0) {
        f1 = RAND_FLT();
    }
    return sqrt(-2.0*log(f1))*cos(TWOPI*RAND_FLT());
}

void initialisation_1d_matrix(int initialisation, float* matrix, int dim, int n_in, int n_out) {
    float lower_bound, distance_bounds;

    if (initialisation == ZERO) {
        for (int i=0; i<dim; i++) {
            matrix[i] = 0;
        }
    } 
    else if (initialisation == XAVIER) 
    {
        lower_bound = -1/sqrt((double)n_in);
        distance_bounds = -2*lower_bound;
        for (int i=0; i < dim; i++) {
            matrix[i] = lower_bound + RAND_FLT()*distance_bounds;
        }
    } 
    else if (initialisation == NORMALIZED_XAVIER) 
    {
        lower_bound = -sqrt(6/(double)(n_in + n_out));
        distance_bounds = -2*lower_bound;
        for (int i=0; i < dim; i++) {
            matrix[i] = lower_bound + RAND_FLT()*distance_bounds;
        }
    } 
    else if (initialisation == HE) 
    {
        distance_bounds = 2/sqrt((double)n_in);
        for (int i=0; i < dim; i++) {
            matrix[i] = randn()*distance_bounds;
        }
    } 
    else 
    {
        printf_warning("Initialisation non reconnue dans 'initialisation_1d_matrix' \n");
    }
}

void initialisation_2d_matrix(int initialisation, float** matrix, int dim1, int dim2, int n_in, int n_out) {
    float lower_bound, distance_bounds;

    if (initialisation == ZERO) {
        for (int i=0; i<dim1; i++) {
            for (int j=0; j<dim2; j++) {
                matrix[i][j] = 0;
            }
        }
    } 
    else if (initialisation == XAVIER) 
    {
        lower_bound = -1/sqrt((double)n_in);
        distance_bounds = -2*lower_bound;
        for (int i=0; i<dim1; i++) {
            for (int j=0; j<dim2; j++) {
                matrix[i][j] = lower_bound + RAND_FLT()*distance_bounds;
            }
        }
    } 
    else if (initialisation == NORMALIZED_XAVIER) 
    {
        lower_bound = -sqrt(6/(double)(n_in + n_out));
        distance_bounds = -2*lower_bound;
        for (int i=0; i<dim1; i++) {
            for (int j=0; j<dim2; j++) {
                matrix[i][j] = lower_bound + RAND_FLT()*distance_bounds;
            }
        }
    } 
    else if (initialisation == HE) 
    {
        distance_bounds = 2/sqrt((double)n_in);
        for (int i=0; i<dim1; i++) {
            for (int j=0; j<dim2; j++) {
                matrix[i][j] = randn()*distance_bounds;
            }
        }
    } 
    else 
    {
        printf_warning("Initialisation non reconnue dans 'initialisation_2d_matrix' \n");
    }
}

void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int dim1, int dim2, int n_in, int n_out) {
    float lower_bound, distance_bounds;

    if (initialisation == ZERO) {
        for (int i=0; i<depth; i++) {
            for (int j=0; j<dim1; j++) {
                for (int k=0; k<dim2; k++) {
                    matrix[i][j][k] = 0;
                }
            }
        }
    } 
    else if (initialisation == XAVIER) 
    {
        lower_bound = -1/sqrt((double)n_in);
        distance_bounds = -2*lower_bound;
        for (int i=0; i<depth; i++) {
            for (int j=0; j<dim1; j++) {
                for (int k=0; k<dim2; k++) {
                    matrix[i][j][k] = lower_bound + RAND_FLT()*distance_bounds;
                }
            }
        }
    } 
    else if (initialisation == NORMALIZED_XAVIER) 
    {
        lower_bound = -sqrt(6/(double)(n_in + n_out));
        distance_bounds = -2*lower_bound;
        for (int i=0; i<depth; i++) {
            for (int j=0; j<dim1; j++) {
                for (int k=0; k<dim2; k++) {
                    matrix[i][j][k] = lower_bound + RAND_FLT()*distance_bounds;
                }
            }
        }
    } 
    else if (initialisation == HE) 
    {
        distance_bounds = 2/sqrt((double)n_in);
        for (int i=0; i<depth; i++) {
            for (int j=0; j<dim1; j++) {
                for (int k=0; k<dim2; k++) {
                    matrix[i][j][k] = randn()*distance_bounds;
                }
            }
        }
    } 
    else 
    {
        printf_warning("Initialisation non reconnue dans 'initialisation_3d_matrix' \n");
    }
}

void initialisation_4d_matrix(int initialisation, float**** matrix, int depth1, int depth2, int dim1, int dim2, int n_in, int n_out) {
    float lower_bound, distance_bounds;

    if (initialisation == ZERO) {
        for (int i=0; i<depth1; i++) {
            for (int j=0; j<depth2; j++) {
                for (int k=0; k<dim1; k++) {
                    for (int l=0; l<depth2; l++) {
                        matrix[i][j][k][l] = 0;
                    }
                }
            }
        }
    } 
    else if (initialisation == XAVIER) 
    {
        lower_bound = -1/sqrt((double)n_in);
        distance_bounds = -2*lower_bound;
        for (int i=0; i<depth1; i++) {
            for (int j=0; j<depth2; j++) {
                for (int k=0; k<dim1; k++) {
                    for (int l=0; l<dim2; l++) {
                        matrix[i][j][k][l] = lower_bound + RAND_FLT()*distance_bounds;
                    }
                }
            }
        }
    } 
    else if (initialisation == NORMALIZED_XAVIER) 
    {
        lower_bound = -sqrt(6/(double)(n_in + n_out));
        distance_bounds = -2*lower_bound;
        for (int i=0; i<depth1; i++) {
            for (int j=0; j<depth2; j++) {
                for (int k=0; k<dim1; k++) {
                    for (int l=0; l<dim2; l++) {
                        matrix[i][j][k][l] = lower_bound + RAND_FLT()*distance_bounds;
                    }
                }
            }
        }
    } 
    else if (initialisation == HE) 
    {
        distance_bounds = 2/sqrt((double)n_in);
        for (int i=0; i<depth1; i++) {
            for (int j=0; j<depth2; j++) {
                for (int k=0; k<dim1; k++) {
                    for (int l=0; l<dim2; l++) {
                        matrix[i][j][k][l] = randn()*distance_bounds;
                    }
                }
            }
        }
    } 
    else 
    {
        printf_warning("Initialisation non reconnue dans 'initialisation_4d_matrix' \n");
    }
}