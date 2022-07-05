#ifndef DEF_STRUCT_H
#define DEF_STRUCT_H

typedef struct Kernel_cnn {
    int k_size;
    int rows;
    int columns;
    int b;
    float*** bias; // De dimension columns*k_size*k_size
    float*** d_bias; // De dimension columns*k_size*k_size
    float**** w; // De dimension rows*columns*k_size*k_size
    float**** d_w; // De dimension rows*columns*k_size*k_size
} Kernel_cnn;

typedef struct Kernel_nn {
    int input_units;
    int output_units;
    float* bias; // De dimension output_units
    float* d_bias; // De dimension output_units
    float** weights; // De dimension input_units*output_units
    float** d_weights; // De dimension input_units*output_units
} Kernel_nn;

typedef struct Kernel {
    Kernel_cnn* cnn;
    Kernel_nn* nn;
    int activation; // Vaut l'activation sauf pour un pooling où il: vaut kernel_size*100 + activation
} Kernel;

typedef struct Layer {

} Layer;

typedef struct Network{
    int dropout; // Contient la probabilité d'abandon entre 0 et 100 (inclus)
    int initialisation; // Contient le type d'initialisation
    int max_size; // Taille maximale du réseau après initialisation
    int size; // Taille actuelle du réseau
    int** dim; // Contient les dimensions de l'input (width*depth)
    Kernel* kernel;
    float**** input;
} Network;

#endif