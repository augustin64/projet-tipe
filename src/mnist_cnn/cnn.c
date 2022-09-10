#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "initialisation.c"
#include "function.c"
#include "creation.c"
#include "make.c"

#include "cnn.h"

// Augmente les dimensions de l'image d'entrée
#define PADDING_INPUT 2

int will_be_drop(int dropout_prob) {
    return (rand() % 100) < dropout_prob;
}

void write_image_in_network_32(int** image, int height, int width, float** input) {
    for (int i=0; i < height+2*PADDING_INPUT; i++) {
        for (int j=PADDING_INPUT; j < width+2*PADDING_INPUT; j++) {
            if (i < PADDING_INPUT || i > height+PADDING_INPUT || j < PADDING_INPUT || j > width+PADDING_INPUT) {
                input[i][j] = 0.;
            }
            else {
                input[i][j] = (float)image[i][j] / 255.0f;
            }
        }
    }
}

void forward_propagation(Network* network) {
    int output_dim, output_depth;
    float*** output;
    for (int i=0; i < network->size-1; i++) {
        if (network->kernel[i]->nn==NULL && network->kernel[i]->cnn!=NULL) { //CNN
            output = network->input[i+1];
            output_dim = network->dim[i+1][0];
            output_depth = network->dim[i+1][1];
            make_convolution(network->input[i], network->kernel[i]->cnn, output, output_dim);
            choose_apply_function_input(network->kernel[i]->activation, output, output_depth, output_dim, output_dim);
        }
        else if (network->kernel[i]->nn!=NULL && network->kernel[i]->cnn==NULL) { //NN
            make_fully_connected(network->input[i][0][0], network->kernel[i]->nn, network->input[i+1][0][0], network->dim[i][0], network->dim[i+1][0]);
            choose_apply_function_input(network->kernel[i]->activation, network->input[i+1], 1, 1, network->dim[i+1][0]);
        }
        else { //Pooling
            if (network->size-2==i) {
                printf("Le réseau ne peut pas finir par une pooling layer");
                return;
            }
            if (network->kernel[i+1]->nn!=NULL && network->kernel[i+1]->cnn==NULL) {
                make_average_pooling_flattened(network->input[i], network->input[i+1][0][0], network->kernel[i]->activation/100, network->dim[i][1], network->dim[i][0]);
                choose_apply_function_input(network->kernel[i]->activation%100, network->input[i+1], 1, 1, network->dim[i+1][0]);
            }
            else if (network->kernel[i+1]->nn==NULL && network->kernel[i+1]->cnn!=NULL) {
                make_average_pooling(network->input[i], network->input[i+1], network->kernel[i]->activation/100, network->dim[i+1][1], network->dim[i+1][0]);
                choose_apply_function_input(network->kernel[i]->activation%100, network->input[i+1], network->dim[i+1][1], network->dim[i+1][0], network->dim[i+1][0]);
            }
            else {
                printf("Le réseau ne peut pas contenir deux pooling layers collées");
                return;
            }
        }
    }
}

void backward_propagation(Network* network, float wanted_number) {
    float* wanted_output = generate_wanted_output(wanted_number);
    int n = network->size-1;
    float loss = compute_cross_entropy_loss(network->input[n][0][0], wanted_output, network->dim[n][0]);
    for (int i=n; i >= 0; i--) {
        if (i==n) {
            if (network->kernel[i]->activation == SOFTMAX) {
                int l2 = network->dim[i][0]; // Taille de la dernière couche
                int l1 = network->dim[i-1][0];
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
    forward_propagation(network);
    return 0;
}