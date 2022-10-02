#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h> // Is it used ?

#include "../colors.h"
#include "include/initialisation.h"
#include "function.c"
#include "creation.c"
#include "make.c"

#include "include/cnn.h"

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
    int activation, input_depth, input_width, output_depth, output_width;
    int n = network->size;
    float*** input;
    float*** output;
    Kernel* k_i;
    for (int i=0; i < n-1; i++) {
        // Transférer les informations de 'input' à 'output'
        k_i = network->kernel[i];
        input = network->input[i];
        input_depth = network->depth[i];
        input_width = network->width[i];
        output = network->input[i+1];
        output_depth = network->depth[i+1];
        output_width = network->width[i+1];
        activation = k_i->activation;

        if (k_i->cnn!=NULL) { // Convolution
            printf("\n(%d)-Convolution of cnn: %dx%dx%d -> %dx%dx%d\n", i, input_depth, input_width, input_width, output_depth, output_width, output_width);
            make_convolution(k_i->cnn, input, output, output_width);
            choose_apply_function_matrix(activation, output, output_depth, output_width);
        }
        else if (k_i->nn!=NULL) { // Full connection
            if (input_depth==1) { // Vecteur -> Vecteur
                printf("\n(%d)-Densification of nn: %dx%dx%d -> %dx%dx%d\n", i, 1, 1, input_width, 1, 1, output_width);
                make_dense(k_i->nn, input[0][0], output[0][0], input_width, output_width);
            } else { // Matrice -> vecteur
                printf("\n(%d)-Densification linearised of nn: %dx%dx%d -> %dx%dx%d\n", i, input_depth, input_width, input_width, 1, 1, output_width);
                make_dense_linearised(k_i->nn, input, output[0][0], input_depth, input_width, output_width);
            }
            choose_apply_function_vector(activation, output, output_width);
        }
        else { // Pooling
            if (n-2==i) {
                printf("Le réseau ne peut pas finir par une pooling layer\n");
                return;
            } else { // Pooling sur une matrice
                printf("\n(%d)-Average pooling: %dx%dx%d -> %dx%dx%d\n", i, input_depth, input_width, input_width, output_depth, output_width, output_width);
                make_average_pooling(input, output, activation/100, output_depth, output_width);
            }
        }
    }
}

void backward_propagation(Network* network, float wanted_number) {
    printf_warning("Appel de backward_propagation, incomplet\n");
    float* wanted_output = generate_wanted_output(wanted_number);
    int n = network->size;
    float loss = compute_cross_entropy_loss(network->input[n][0][0], wanted_output, network->width[n]);
    int activation, input_depth, input_width, output_depth, output_width;
    float*** input;
    float*** output;
    Kernel* k_i;
    Kernel* k_i_1;

    for (int i=n-3; i >= 0; i--) {
        // Modifie 'k_i' à partir d'une comparaison d'informations entre 'input' et 'output'
        k_i = network->kernel[i];
        k_i_1 = network->kernel[i+1];
        input = network->input[i];
        input_depth = network->depth[i];
        input_width = network->width[i];
        output = network->input[i+1];
        output_depth = network->depth[i+1];
        output_width = network->width[i+1];
        activation = k_i->activation;

        //if convolution
        // else if dense (linearised or not)
        // else pooling
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