#include <stdio.h>
#include "include/print.h"

#define print_bar printf("---------------------------\n")
#define print_space printf("\n")
#define print_dspace printf("\n\n")
#define print_tspace printf("\n\n\n")
#define green printf("\033[0;31m")
#define red printf("\033[0;32m")
#define blue printf("\033[0;34m")
#define purple printf("\033[0;35m")
#define reset_color printf("\033[0m")

void print_kernel_cnn(Kernel_cnn* k, int depth_input, int dim_input, int depth_output, int dim_output) {
    int k_size = dim_input - dim_output + 1;
    // print bias
    green;
    for (int i=0; i<depth_output; i++) {
        for (int j=0; j<dim_output; j++) {
            for (int k=0; k<dim_output; k++) {
                printf("%.2f", k->bias[i][j][k]);
            }
            print_space;
        }
        print_dspace;
    }
    print_dspace;
    reset_color;

    //print weights
    red;
    for (int i=0; i<; i++) {
        printf("------Line %d-----\n", i);
        for (int j=0; j<; j++) {
            for (int k=0; k<; k++) {
                for (int l=0; l<; l++) {
                    printf("%.2f", k->w[i][j][k][l]);
                }
                print_space;
            }
            print_dspace;
        }
        print_dspace;
    }
    reset_color;
    print_dspace;

}

void print_pooling(int size) {
    print_bar;
    purple;
    printf("-------Pooling %dx%d-------\n", size ,size);
    reset_color;
    print_bar;
    print_dspace;
}

void print_kernel_nn(Kernel_nn* k, int size_input, int size_output) {
    // print bias
    green;
    for (int i=0; i<size_output; i++) {
        printf("%.2f ", k->bias[i]);
    }
    print_dspace;
    reset_color;

    //print weights
    red;
    for (int i=0; i<size_output; i++) {
        for (int j=0; j<size_input; j++) {
            printf("%.2f ", k->weights[j][i]);
        }
        print_space;
    }
    reset_color;
    print_dspace;
}

void print_input(float*** input, int depth, int dim) {
    print_bar;
    print_bar;
    blue;
    for (int i=0; i<depth; i++) {
        for (int j=0; j<dim; j++) {
            for (int k=0; k<dim; k++) {
                printf("%.2f ", input[i][j][k]);
            }
            print_space;
        }
        print_dspace;
    }
    reset_color;
    print_dspace;
}

void print_cnn(Network* network) {
    int n = network->size;
    int input_depth, input_width, output_depth, output_width;
    Kernel* k_i;
    for (int i=0; i<(n-1); i++) {
        input = network->input[i];
        input_depth = network->depth[i];
        input_width = network->width[i];
        output = network->input[i+1];
        output_depth = network->depth[i+1];
        output_width = network->width[i+1];
        k_i = network->kernel[i];

        print_input(input, input_depth, input_width);

        if (k_i->cnn) { // Convolution
            print_kernel_cnn(k_i->cnn, input_depth, input_width, output_depth, output_width);
        }
        else if (k_i->nn) { // Full connection
            print_kernel_nn(k_i->nn, input_width, output_width);
        }
        else { // Pooling
            print_pooling(input_width - output_width +1);
        }
    }
    print_input(input[n-1], network->depth[n-1], network->width[n-1]);
}