#include <stdlib.h>
#include <stdio.h>

#include "include/creation.h"
#include "include/function.h"
#include "include/struct.h"

#include "include/models.h"

Network* create_network_lenet5(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth, int finetuning) {
    Network* network = create_network(8, learning_rate, dropout, initialisation, input_width, input_depth, finetuning);
    add_convolution(network, 5, 6, 1, 0, activation);
    add_average_pooling(network, 2, 2, 0);
    add_convolution(network, 5, 16, 1, 0, activation);
    add_average_pooling(network, 2, 2, 0);
    add_dense_linearisation(network, 120, activation);
    add_dense(network, 84, activation);
    add_dense(network, 10, SOFTMAX);
    return network;
}

Network* create_network_alexnet(float learning_rate, int dropout, int activation, int initialisation, int size_output, int finetuning) {
    Network* network = create_network(12, learning_rate, dropout, initialisation, 227, 3, finetuning);
    add_convolution(network, 11, 96, 4, 0, activation);
    add_average_pooling(network, 3, 2, 0);
    add_convolution(network, 5, 256, 1, 2, activation);
    add_average_pooling(network, 3, 2, 0);
    add_convolution(network, 3, 384, 1, 1, activation);
    add_convolution(network, 3, 384, 1, 1, activation);
    add_convolution(network, 3, 256, 1, 1, activation);
    add_average_pooling(network, 3, 2, 0);
    add_dense_linearisation(network, 4096, activation);
    add_dense(network, 4096, activation);
    add_dense(network, size_output, SOFTMAX);
    return network;
}

Network* create_network_VGG16(float learning_rate, int dropout, int activation, int initialisation, int size_output, int finetuning) {
    Network* network = create_network(22, learning_rate, dropout, initialisation, 256, 3, finetuning);
    add_convolution(network, 3, 64, 1, 1, activation); // Conv3-64
    add_convolution(network, 3, 64, 1, 1, activation); // Conv3-64
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 128, 1, 1, activation); // Conv3-128
    add_convolution(network, 1, 128, 1, 0, activation); // Conv1-128
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 256, 1, 1, activation); // Conv3-256
    add_convolution(network, 3, 256, 1, 1, activation); // Conv3-256
    add_convolution(network, 1, 256, 1, 0, activation); // Conv1-256
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 1, 512, 1, 0, activation); // Conv1-512
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 1, 512, 1, 0, activation); // Conv1-512
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_dense_linearisation(network, 4096, activation);
    add_dense(network, 4096, activation);
    add_dense(network, size_output, SOFTMAX);
    return network;
}

Network* create_network_VGG16_227(float learning_rate, int dropout, int activation, int initialisation, int finetuning) {
    Network* network = create_network(22, learning_rate, dropout, initialisation, 227, 3, finetuning);
    add_convolution(network, 3, 64, 1, 1, activation); // Conv3-64
    add_convolution(network, 3, 64, 1, 1, activation); // Conv3-64
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 128, 1, 1, activation); // Conv3-128
    add_convolution(network, 1, 128, 1, 0, activation); // Conv1-128
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 256, 1, 1, activation); // Conv3-256
    add_convolution(network, 3, 256, 1, 1, activation); // Conv3-256
    add_convolution(network, 1, 256, 1, 0, activation); // Conv1-256
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 1, 512, 1, 0, activation); // Conv1-512
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 3, 512, 1, 1, activation); // Conv3-512
    add_convolution(network, 1, 512, 1, 0, activation); // Conv1-512
    add_average_pooling(network, 2, 2, 0); // Max Pool

    add_dense_linearisation(network, 4096, activation);
    add_dense(network, 4096, activation);
    add_dense(network, 1000, SOFTMAX);
    return network;
}

Network* create_simple_one(float learning_rate, int dropout, int activation, int initialisation, int input_width, int input_depth, int finetuning) {
    Network* network = create_network(3, learning_rate, dropout, initialisation, input_width, input_depth, finetuning);
    add_dense_linearisation(network, 80, activation);
    add_dense(network, 10, SOFTMAX);
    return network;
}