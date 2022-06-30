#include <stdlib.h>
#include <stdio.h>
#include <time.h>
typedef struct Neuron{
    float bias; // Caractérise le bias du neurone
    float z; // Sauvegarde des calculs faits sur le neurone (programmation dynamique)

    float back_bias; // Changement du bias lors de la backpropagation
    float last_back_bias; // Dernier changement de back_bias
} Neuron;

typedef struct Bias{
    float bias; 
    float back_bias;
    float last_back_bias;
} Bias;

typedef struct Matrix {
    int rows; // Nombre de lignes de la matrice
    int columns; // Nombre de colonnes de la matrice
    float** value; // Tableau 2d comportant les valeurs de matrice
} Matrix;

typedef struct Matrix_of_neurons {
    int rows; // Nombre de lignes de la matrice
    int columns; // Nombre de colonnes de la matrice
    float** Neuron; // Tableau 2d comportant les valeurs de matrice
} Matrix_of_neurons;

typedef struct Matrix_of_bias {
    int rows;
    int columns;
    float*** bias;
} Matrix_of_bias;

typedef struct Layer {
    int rows; // Nombre de matrices du tableau de neurones
    Matrix** conv; // Tableau de matrices des neurones dans la couche
} Layer;

typedef struct Filter {
    int columns;
    int rows;
    int dim_bias;
    Matrix_of_neurons*** kernel; // De dimension columns*rows
    Matrix_of_bias** bias; // De dimension columns
} Filter;

typedef struct Network{
    int dropout; // Contains the probability of dropout bewteen 0 and 100
    int max_size;
    int size; // Taille total du réseau
    int size_cnn; // Nombre de couches dans le cnn
    int* type_kernel; //De taille size -1

    Layer** input; // Tableau des couches dans le réseau neuronal
    Filter** kernel;
} Network;

void write_image_in_newtork_32(int** image, int height, int width, float** network) {
    /* Ecrit une image 28*28 au centre d'un tableau 32*32 et met à 0 le reste */

    for (int i=0; i < height+2*PADING_INPUT; i++) {
        for (int j=PADING_INPUT; j < width+2*PADING_INPUT; j++) {
            if (i<PADING_INPUT || i>height+PADING_INPUT || j<PADING_INPUT || j>width+PADING_INPUT){
                network[i][j] = 0.;
            }
            else {
                network[i][j] = (float)image[i][j] / 255.0f;
            }
        }
    }
}



void make_convolution(Layer* input, Filter* filter, Layer* output){
    /* Effectue une convolution sans stride */
    if (filter->columns != output->rows) {
        printf("Erreur, le filtre de la convolution et la sortie ne sont pas compatibles");
        return;
    }
    if (filter->dim_bias != output->rows) {
        printf("Erreur, le biais et la sortie de la convolution n'ont pas les mêmes dimensions");
        return;
    }

    // MISS CONDITIONS ON THE CONVOLUTION
    int i, j, k;
    for (i=0; i < filter->rows; i++) {
        for (j=0; j < filter->dim_bias; j++) {
            for (int k=0; k < filter->dim_bias; k++) {
                //output->conv[j][k] = filter->bias[i]->bias
                // COPY BIAS OF FILTERS IN OUTPUT
                // POUR CHAQUE COLONNE DANS LE KERNEL
                    // ON APPLIQUE LE FILTRE SUR CHAQUE LIGNE DE L'INPUT ET LES SOMMES
            }
        }
    }
}

void make_average_pooling(Layer* input, int dim_pooling, Layer* output){
    /* Effectue un average pooling avec full strides */
    
    if (input->rows != output->rows || output->conv[0]->rows*dim_pooling != input->conv[0]->rows || input->rows != output->rows) {
        printf("Erreur, dimension de la sortie et de l'entrée ne sont pas compatibles avec l'average pooling");
        return;
    }
    int i, j, k, a, b, nb=dim_pooling*dim_pooling;
    for (i=0; i < input->rows; i++) {
        for (j=0; j < output->conv[i]->rows; j++) {
            for (k=0; k < output->conv[i]->columns; k++) {
                output->conv[i]->value[j][k] = 0;
                for (a=0; a < dim_pooling; a++) {
                    for (b=0; b < dim_pooling; b++) {
                        output->conv[i]->value[j][k] += input->conv[i]->value[dim_pooling*j + a][dim_pooling*k + b];
                    }
                }
                output->conv[i]->value[j][k] /= nb;
            }
        }
    }
}

void forward_propagation_cnn() {
    /* Effectue une forward propagation d'un cnn */
}