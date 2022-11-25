#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "../mnist/include/mnist.h"
#include "include/neuron_io.h"
#include "include/struct.h"
#include "include/jpeg.h"
#include "include/free.h"
#include "include/cnn.h"

void test_network(int dataset_type, char* modele, char* images_file, char* labels_file, char* data_dir, bool preview_fails) {

}


void recognize_mnist(Network* network, char* input_file) {
    int width, height; // Dimensions de l'image
    int nb_elem; // Nombre d'éléments
    int maxi; // Catégorie reconnue

    // Load image
    int* mnist_parameters = read_mnist_images_parameters(input_file);
    int*** images = read_mnist_images(input_file);
    nb_elem = mnist_parameters[0];

    width = mnist_parameters[1];
    height = mnist_parameters[2];
    free(mnist_parameters);

    printf("Image\tCatégorie détectée\n");
    // Load image in the first layer of the Network
    for (int i=0; i < nb_elem; i++) {
        write_image_in_network_32(images[i], height, width, network->input[0][0]);
        forward_propagation(network);
        maxi = indice_max(network->input[network->size-1][0][0], 10);

        printf("%d\t%d\n", i, maxi);

        for (int j=0; j < height; j++) {
            free(images[i][j]);
        }
        free(images[i]);
    }
    free(images);
}

void recognize_jpg(Network* network, char* input_file) {
    int width, height; // Dimensions de l'image
    int maxi;

    imgRawImage* image = loadJpegImageFile(input_file);
    width = image->width;
    height = image->height;

    write_image_in_network_260(image->lpData, height, width, network->input[0]);
    forward_propagation(network);
    maxi = indice_max(network->input[network->size-1][0][0], 50);

    printf("Catégorie reconnue: %d\n", maxi);
    free(image->lpData);
    free(image);
}

void recognize(int dataset_type, char* modele, char* input_file) {
    Network* network = read_network(modele);

    if (dataset_type == 0) {
        recognize_mnist(network, input_file);
    } else {
        recognize_jpg(network, input_file);
    }

    free_network(network);
}