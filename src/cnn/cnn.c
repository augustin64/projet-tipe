#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <float.h> // Is it used ?
#include <math.h>

#include "include/backpropagation.h"
#include "include/initialisation.h"
#include "include/function.h"
#include "include/creation.h"
#include "include/update.h"
#include "include/make.h"

#include "../include/colors.h"
#include "include/cnn.h"

// Augmente les dimensions de l'image d'entrée
#define PADDING_INPUT 2

int indice_max(float* tab, int n) {
    int indice = -1;
    float maxi = -FLT_MAX;

    for (int i=0; i < n; i++) {
        if (tab[i] > maxi) {
            maxi = tab[i];
            indice = i;
        }
    }
    return indice;
}

int will_be_drop(int dropout_prob) {
    return (rand() % 100) < dropout_prob;
}

void write_image_in_network_32(int** image, int height, int width, float** input, bool random_offset) {
    int i_offset = 0;
    int j_offset = 0;
    int min_col = 0;
    int min_ligne = 0;

    if (random_offset) {
        /*
                                    <-- min_ligne
                .%%:.
                ######%%%%%%%%%.
                .:.:%##########:
                        . .... ##:
                            .##
                            ##.
                            :##
                            .##.
                            :#%
                            %#.
                        :#%
                        .##.
                        ##%
                        %##
                        ##.
                        ##:
                    :##.
                    .###.
                    :###
                    :#%
                                    <-- max_ligne
               ^-- min_col
                                 ^-- max_col
        */
        int sum_colonne[width];
        int sum_ligne[height];

        for (int i=0; i < width; i++) {
            sum_colonne[i] = 0;
        }
        for (int j=0; j < height; j++) {
            sum_ligne[j] = 0;
        }

        for (int i=0; i < width; i++) {
            for (int j=0; j < height; j++) {
                sum_ligne[i] += image[i][j];
                sum_colonne[j] += image[i][j];
            }
        }

        min_ligne = -1;
        while (sum_ligne[min_ligne+1] == 0 && min_ligne < width+1) {
            min_ligne++;
        }

        int max_ligne = width;
        while (sum_ligne[max_ligne-1] == 0 && max_ligne > 0) {
            max_ligne--;
        }

        min_col = -1;
        while (sum_colonne[min_col+1] == 0 && min_col < height+1) {
            min_col++;
        }

        int max_col = height;
        while (sum_colonne[max_col-1] == 0 && max_col > 0) {
            max_col--;
        }

        i_offset = rand()%(27-max_ligne+min_ligne);
        j_offset = rand()%(27-max_col+min_col);
    }

    int padding = (32 - height)/2;
    for (int i=0; i < padding; i++) {
        for (int j=0; j < 32; j++) {
            input[i][j] = 0.;
            input[31-i][j] = 0.;
            input[j][i] = 0.;
            input[j][31-i] = 0.;
        }
    }

    for (int i=0; i < width; i++) {
        for (int j=0; j < height; j++) {
            int adjusted_i = i + min_ligne - i_offset;
            int adjusted_j = j + min_col - j_offset;
            // Make sure not to be out of the image
            input[i+2][j+2] = adjusted_i < height && adjusted_j < width && adjusted_i >= 0 && adjusted_j >= 0 ? (float)image[adjusted_i][adjusted_j] / 255.0f : 0.;
        }
    }
}

void write_image_in_network_260(unsigned char* image, int height, int width, float*** input) {
    int size_input = 260;
    int padding = (size_input - height)/2;

    for (int i=0; i < padding; i++) {
        for (int j=0; j < size_input; j++) {
            for (int composante=0; composante < 3; composante++) {
                input[composante][i][j] = 0.;
                input[composante][size_input-1-i][j] = 0.;
                input[composante][j][i] = 0.;
                input[composante][j][size_input-1-i] = 0.;
            }
        }
    }

    for (int i=0; i < width; i++) {
        for (int j=0; j < height; j++) {
            for (int composante=0; composante < 3; composante++) {
                input[composante][i+2][j+2] = (float)image[(i*height+j)*3 + composante] / 255.0f;
            }
        }
    }
}

void forward_propagation(Network* network) {
    int n = network->size; // Nombre de couches du réseau, il contient n-1 kernels

    for (int i=0; i < n-1; i++) {
        /*
        * On procède kernel par kernel:
        * On considère à chaque fois une couche d'entrée, une couche de sortie et le kernel qui contient les informations
        * pour passer d'une couche à l'autre
        */
        Kernel* k_i = network->kernel[i];


        float*** input = network->input[i]; // Couche d'entrée
        int input_depth = network->depth[i]; // Dimensions de la couche d'entrée
        int input_width = network->width[i];

        float*** output_z = network->input_z[i+1]; // Couche de sortie avant que la fonction d'activation ne lui soit appliquée
        float*** output = network->input[i+1]; // Couche de sortie
        int output_depth = network->depth[i+1]; // Dimensions de la couche de sortie
        int output_width = network->width[i+1];

        int activation = k_i->activation;
        int pooling = k_i->pooling;

        if (k_i->nn) {
            drop_neurones(input, 1, 1, input_width, network->dropout);
        } else {
            drop_neurones(input, input_depth, input_width, input_width, network->dropout);
        }

        /*
        * Pour chaque couche excepté le pooling, on propage les valeurs de la couche précédente,
        * On copie les valeurs de output dans output_z, puis on applique la fonction d'activation à output_z
        */
        if (k_i->cnn) { // Convolution
            make_convolution(k_i->cnn, input, output, output_width);
            copy_input_to_input_z(output, output_z, output_depth, output_width, output_width);
            apply_function_to_matrix(activation, output, output_depth, output_width);
        }
        else if (k_i->nn) { // Full connection
            if (k_i->linearisation == DOESNT_LINEARISE) { // Vecteur -> Vecteur
                make_dense(k_i->nn, input[0][0], output[0][0], input_width, output_width);
            } else { // Matrice -> Vecteur
                make_dense_linearized(k_i->nn, input, output[0][0], input_depth, input_width, output_width);
            }
            copy_input_to_input_z(output, output_z, 1, 1, output_width);
            apply_function_to_vector(activation, output, output_width);
        }
        else { // Pooling
            if (i == n-2) {
                printf_error("Le réseau ne peut pas finir par un pooling layer\n");
                return;
            } else { // Pooling sur une matrice
                if (pooling == AVG_POOLING) {
                    make_average_pooling(input, output, input_width/output_width, output_depth, output_width);
                } else if (pooling == MAX_POOLING) {
                    make_max_pooling(input, output, input_width/output_width, output_depth, output_width);
                } else {
                    printf_error("Impossible de reconnaître le type de couche de pooling: ");
                    printf("identifiant: %d, position: %d\n", pooling, i);
                }
            }
            copy_input_to_input_z(output, output_z, output_depth, output_width, output_width);
        }
    }
}

void backward_propagation(Network* network, int wanted_number) {
    int n = network->size; // Nombre de couches du réseau

    // Backward sur la dernière couche qui utilise toujours SOFTMAX
    float* wanted_output = generate_wanted_output(wanted_number, network->width[network->size -1]); // Sortie désirée, permet d'initialiser une erreur
    softmax_backward_cross_entropy(network->input[n-1][0][0], wanted_output, network->width[n-1]);
    free(wanted_output);

    /*
    * On propage à chaque étape:
    * - les dérivées de l'erreur par rapport aux poids et biais, que l'on ajoute à ceux existants dans kernel->_->d_bias/d_weights
    * - les dérivées de l'erreur par rapport à chaque case de input, qui servent uniquement à la propagation des informations.
    * Ainsi, on écrase les valeurs contenues dans input, mais on utilise celles restantes dans input_z qui indiquent les valeurs avant
    * la composition par la fonction d'activation pour pouvoir continuer à remonter.
    */
    for (int i=n-2; i >= 0; i--) {
        // Modifie 'k_i' à partir d'une comparaison d'informations entre 'input' et 'output'
        Kernel* k_i = network->kernel[i];

        float*** input = network->input[i];
        float*** input_z = network->input_z[i];
        int input_depth = network->depth[i];
        int input_width = network->width[i];

        float*** output = network->input[i+1];
        int output_depth = network->depth[i+1];
        int output_width = network->width[i+1];

        int activation = i==0?SIGMOID:network->kernel[i-1]->activation;


        if (k_i->cnn) { // Convolution
            funcPtr d_f = get_activation_function(-activation);
            backward_convolution(k_i->cnn, input, input_z, output, input_depth, input_width, output_depth, output_width, d_f, i==0);
        } else if (k_i->nn) { // Full connection
            funcPtr d_f = get_activation_function(-activation);
            if (k_i->linearisation == DOESNT_LINEARISE) { // Vecteur -> Vecteur
                backward_dense(k_i->nn, input[0][0], input_z[0][0], output[0][0], input_width, output_width, d_f, i==0);
            } else { // Matrice -> vecteur
                backward_linearisation(k_i->nn, input, input_z, output[0][0], input_depth, input_width, output_width, d_f);
            }
        } else { // Pooling
            if (k_i->pooling == AVG_POOLING) {
                backward_average_pooling(input, output, input_width, output_width, input_depth); // Depth pour input et output a la même valeur
            } else {
                backward_max_pooling(input, output, input_width, output_width, input_depth); // Depth pour input et output a la même valeur
            }
        }
    }
}

void drop_neurones(float*** input, int depth, int dim1, int dim2, int dropout) {
    for (int i=0; i < depth; i++) {
        for (int j=0; j < dim1; j++) {
            for (int k=0; k < dim2; k++) {
                if (will_be_drop(dropout))
                    input[i][j][k] = 0;
            }
        }
    }
}

void copy_input_to_input_z(float*** output, float*** output_z, int output_depth, int output_rows, int output_columns) {
    for (int i=0; i<output_depth; i++) {
        for (int j=0; j<output_rows; j++) {
            for (int k=0; k<output_columns; k++) {
                output_z[i][j][k] = output[i][j][k];
            }
        }
    }
}

float compute_mean_squared_error(float* output, float* wanted_output, int len) {
    /*
    * $E = \frac{ \sum_{i=0}^n (output_i - desired output_i)^2 }{n}$ 
    */
    if (len == 0) {
        printf_error("MSE: division par 0\n");
        return 0.;
    }
    float loss=0.;
    for (int i=0; i < len ; i++) {
        loss += (output[i]-wanted_output[i])*(output[i]-wanted_output[i]);
    }
    return loss/len;
}

float compute_cross_entropy_loss(float* output, float* wanted_output, int len) {
    float loss=0.;
    for (int i=0; i < len ; i++) {
        if (wanted_output[i]==1) {
            if (output[i]==0.) {
                loss -= log(FLT_EPSILON);
            }
            else {
                loss -= log(output[i]);
            }
        }
    }
    return loss/len;
}

float* generate_wanted_output(int wanted_number, int size_output) {
    float* wanted_output = (float*)malloc(sizeof(float)*size_output);
    for (int i=0; i < size_output; i++) {
        if (i==wanted_number) {
            wanted_output[i]=1;
        }
        else {
            wanted_output[i]=0;
        }
    }
    return wanted_output;
}