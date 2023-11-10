#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

#include "include/test.h"

#include "../src/common/include/colors.h"
#include "../src/common/include/mnist.h"


void read_test(int nb_images, int width, int height, int*** images, unsigned int* labels) {
    for (int i=0; i < nb_images; i++) {
        (void)labels[i];
    }
    _TEST_ASSERT(true, "Accès en lecture des labels");
    
    for (int i=0; i < nb_images; i++) {
        for (int j=0; j < height; j++) {
            for (int k=0; k < width; k++) {
                (void)images[i][j][k];
            }
        }
    }
    _TEST_ASSERT(true, "Accès en lecture des images");
}

int main() {
    _TEST_PRESENTATION("Mnist: Accès en lecture");

    char* image_file = (char*)"data/mnist/t10k-images-idx3-ubyte";
    char* labels_file = (char*)"data/mnist/t10k-labels-idx1-ubyte";

    int* parameters = read_mnist_images_parameters(image_file);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    _TEST_ASSERT(true, "Chargement des paramètres");

    int*** images = read_mnist_images(image_file);
    _TEST_ASSERT(true, "Chargement des images");

    unsigned int* labels = read_mnist_labels(labels_file);
    _TEST_ASSERT(true, "Chargement des labels");

    read_test(nb_images, width, height, images, labels);

    for (int i=0; i < nb_images; i++) {
        for (int j=0; j < height; j++) {
            free(images[i][j]);
        }
        free(images[i]);
    }
    free(images);
    free(labels);
    free(parameters);
    return 0;
}