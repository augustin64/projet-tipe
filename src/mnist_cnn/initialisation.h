#ifndef DEF_INITIALISATION_H
#define DEF_INITIALISATION_H

// Génère un flotant entre 0 et 1
#define RAND_FLT() ((float)rand())/((float)RAND_MAX)

#define ZERO 0
#define GLOROT_NORMAL 1
#define GLOROT_UNIFROM 2
#define HE_NORMAL 3
#define HE_UNIFORM 4

/*
* Initialise une matrice 1d rows de float en fonction du type d'initialisation
*/
void initialisation_1d_matrix(int initialisation, float* matrix, int rows, int n); //NOT FINISHED (UNIFORM AND VARIATIONS)

/*
* Initialise une matrice 2d rows*columns de float en fonction du type d'initialisation
*/
void initialisation_2d_matrix(int initialisation, float** matrix, int rows, int columns, int n); //NOT FINISHED

/*
* Initialise une matrice 3d depth*dim*columns de float en fonction du type d'initialisation
*/
void initialisation_3d_matrix(int initialisation, float*** matrix, int depth, int rows, int columns, int n); //NOT FINISHED

/*
* Initialise une matrice 4d rows*columns*rows1*columns1 de float en fonction du type d'initialisation
*/
void initialisation_4d_matrix(int initialisation, float**** matrix, int rows, int columns, int rows1, int columns1, int n); //NOT FINISHED

#endif