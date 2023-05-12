#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

#include "../src/common/include/colors.h"
#include "../src/common/include/mnist.h"


void read_test(int nb_images, int width, int height, int*** images, unsigned int* labels) {
    printf("\tLecture des labels\n");
    for (int i=0; i < nb_images; i++) {
        (void)labels[i];
    }
    printf(GREEN "\tOK\n" RESET);
    printf("\tLecture des images\n");
    for (int i=0; i < nb_images; i++) {
        for (int j=0; j < height; j++) {
            for (int k=0; k < width; k++) {
                (void)images[i][j][k];
            }
        }
    }
    printf(GREEN "\tOK\n" RESET);
}

int main() {
    char* image_file = (char*)"data/mnist/t10k-images-idx3-ubyte";
    char* labels_file = (char*)"data/mnist/t10k-labels-idx1-ubyte";
    printf("Chargement des paramètres\n");

    int* parameters = read_mnist_images_parameters(image_file);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    printf(GREEN "OK\n" RESET);
    printf("Chargement des images\n");

    int*** images = read_mnist_images(image_file);

    printf(GREEN "OK\n" RESET);
    printf("Chargement des labels\n");

    unsigned int* labels = read_mnist_labels(labels_file);

    printf(GREEN "OK\n" RESET);
    printf("Vérification de l'accès en lecture\n");

    read_test(nb_images, width, height, images, labels);

    printf(GREEN "OK\n" RESET);
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