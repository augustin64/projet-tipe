#ifndef DEF_STRUCT_H
#define DEF_STRUCT_H

typedef struct Kernel_cnn {
    int k_size;
    int rows; // Depth of the input
    int columns; // Depth of the output
    float*** bias; // bias[columns][k_size][k_size]
    float*** d_bias; // d_bias[columns][k_size][k_size]
    float**** w; // w[rows][columns][k_size][k_size]
    float**** d_w; // dw[rows][columns][k_size][k_size]
} Kernel_cnn;

typedef struct Kernel_nn {
    int input_units; // Nombre d'éléments en entrée
    int output_units; // Nombre d'éléments en sortie
    float* bias; // bias[output_units]
    float* d_bias; // d_bias[output_units]
    float** weights; // weight[input_units][output_units]
    float** d_weights; // d_weights[input_units][output_units]
} Kernel_nn;

typedef struct Kernel {
    Kernel_cnn* cnn; // NULL si ce n'est pas un cnn
    Kernel_nn* nn; // NULL si ce n'est pas un nn
    int activation; // Vaut l'activation sauf pour un pooling où il: vaut pooling_size*100 + activation
    int linearisation; // Vaut 1 si c'est la linéarisation d'une couche, 0 sinon ?? Ajouter dans les autres
} Kernel;


typedef struct Network{
    int dropout; // Contient la probabilité d'abandon d'un neurone dans [0, 100] (entiers)
    int initialisation; // Contient le type d'initialisation
    int max_size; // Taille du tableau contenant le réseau
    int size; // Taille actuelle du réseau (size ≤ max_size)
    int* width; // width[size]
    int* depth; // depth[size]
    Kernel** kernel; // Tableau de tous les kernels
    float**** input; // Tableau de toutes les couches du réseau input[nb couches][?][?][?]
} Network;

#endif