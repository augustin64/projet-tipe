#ifndef DEF_INITIALISATION_H
#define DEF_INITIALISATION_H

// Génère un flottant entre 0 et 1
#define RAND_FLT() ((float)rand())/((float)RAND_MAX)

#define ZERO 0
#define GLOROT 1
#define XAVIER 1 // Xavier and Glorot initialisations are the same
#define HE 2

/*
* Initialise une matrice 1d dim de float en fonction du type d'initialisation
*/
void initialisation_1d_matrix(int initialisation, float* matrix, int dim, int n_in);

/*
* Initialise une matrice 2d dim1*dim2 de float en fonction du type d'initialisation
*/
void initialisation_2d_matrix(int initialisation, float** matrix, int dim1, int dim2, int n_in, int n_out);

/*
* Initialise une matrice 3d depth*dim1*dim2 de float en fonction du type d'initialisation
*/
void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int dim1, int dim2, int n_in, int n_out);

/*
* Initialise une matrice 4d depth1*depth2*dim1*dim2 de float en fonction du type d'initialisation
*/
void initialisation_4d_matrix(int initialisation, float**** matrix, int depth1, int depth2, int dim1, int dim2, int n_in, int n_out);

#endif