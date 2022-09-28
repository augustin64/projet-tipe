#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

#include "../colors.h"
#include "include/initialisation.h"
#include "function.c"
#include "creation.c"
#include "make.c"

#include "include/main.h"

// Augmente les dimensions de l'image d'entrée
#define PADDING_INPUT 2

int will_be_drop(int dropout_prob) {
    return (rand() % 100) < dropout_prob;
}

void write_image_in_network_32(int** image, int height, int width, float** input) {
    for (int i=0; i < height+2*PADDING_INPUT; i++) {
        for (int j=0; j < width+2*PADDING_INPUT; j++) {
            if (i < PADDING_INPUT || i >= height+PADDING_INPUT || j < PADDING_INPUT || j >= width+PADDING_INPUT) {
                input[i][j] = 0.;
            }
            else {
                input[i][j] = (float)image[i][j] / 255.0f;
            }
        }
    }
}

void forward_propagation(Network* network) {
    int activation, input_width, input_depth, output_width, output_depth;
    int n = network->size;
    float*** input;
    float*** output;
    Kernel* k_i_1;
    Kernel* k_i;
    for (int i=0; i < n-1; i++) {
        k_i_1 = network->kernel[i+1];
        k_i = network->kernel[i];
        input_width = network->width[i];
        input_depth = network->depth[i];
        output_width = network->width[i+1];
        output_depth = network->depth[i+1];
        activation = network->kernel[i]->activation;
        input = network->input[i];
        output = network->input[i+1];

        if (k_i_1->nn==NULL && k_i_1->cnn!=NULL) { //CNN
            printf("Convolution of cnn: %dx%d -> %dx%d\n", input_depth, input_width, output_depth, output_width);
            make_convolution(input, k_i_1->cnn, output, output_width);
            choose_apply_function_input(activation, output, output_depth, output_width, output_width);
        }
        else if (k_i_1->nn!=NULL && k_i_1->cnn==NULL) { //NN
            printf("Densification of nn\n");
            // Checked if it is a nn which linearise
            make_fully_connected(network->input[i][0][0], network->kernel[i]->nn, network->input[i+1][0][0], input_width, output_width);
            choose_apply_function_input(activation, output, 1, 1, output_width);
        }
        else { //Pooling (Vérifier dedans) ??
            if (n-2==i) {
                printf("Le réseau ne peut pas finir par une pooling layer");
                return;
            }
            if (1==1) { // Pooling sur une matrice
                printf("Average pooling\n");
                make_average_pooling(input, output, activation/100, output_depth, output_width);
            }
            else if (1==0) { // Pooling sur un vecteur
                printf("Error: Not implemented: forward: %d\n", i);
            }
            else {
                printf("Erreur: forward_propagation: %d -> %d %d\n", i, k_i_1->nn==NULL, k_i_1->cnn);
                return;
            }
        }
    }
}

void backward_propagation(Network* network, float wanted_number) { // TODO
    printf_warning("Appel de backward_propagation, incomplet\n");
    float* wanted_output = generate_wanted_output(wanted_number);
    int n = network->size-1;
    float loss = compute_cross_entropy_loss(network->input[n][0][0], wanted_output, network->width[n]);
    for (int i=n; i >= 0; i--) {
        if (i==n) {
            if (network->kernel[i]->activation == SOFTMAX) {
                int l2 = network->width[i]; // Taille de la dernière couche
                int l1 = network->width[i-1];
                for (int j=0; j < l2; j++) {

                }
            }
            else {
                printf("Erreur, seule la fonction SOFTMAX est implémentée pour la dernière couche");
                return;
            }
        }
        else {
            if (network->kernel[i]->activation == SIGMOID) {

            }
            else if (network->kernel[i]->activation == TANH) {

            }
            else if (network->kernel[i]->activation == RELU) {
                
            }
        }
    }
    free(wanted_output);
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
    return loss;
}
                
float* generate_wanted_output(float wanted_number) {
    float* wanted_output = (float*)malloc(sizeof(float)*10);
    for (int i=0; i < 10; i++) {
        if (i==wanted_number) {
            wanted_output[i]=1;
        }
        else {
            wanted_output[i]=0;
        }
    }
    return wanted_output;
}

int main() {
    Network* network = create_network_lenet5(0, TANH, GLOROT_NORMAL);
    for (int i=0; i<8; i++) {
        printf("%d %d \n", network->depth[i], network->width[i]);
    }
    printf("Kernel:\n");
    for (int i=0; i<7; i++) {
        if (network->kernel[i]->cnn!=NULL) {
            printf("%d -> %d %d\n", i, network->kernel[i]->cnn->rows, network->kernel[i]->cnn->k_size);
        }
    }
    forward_propagation(network);
    return 0;
}