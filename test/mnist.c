#include <stdlib.h>
#include <stdio.h>
#include <inttypes.h>

#include "../src/mnist/mnist.c"


void test_lecture(int nb_images, int width, int height, int*** images, unsigned int* labels) {
    printf("\tLecture des labels\n");
    for (int i=0; i < nb_images; i++) {
        (void)labels[i];
    }
    printf("\tOK\n");
    printf("\tLecture des images\n");
    for (int i=0; i < nb_images; i++) {
        for (int j=0; j < height; j++) {
            for (int k=0; k < width; k++) {
                (void)images[i][j][k];
            }
        }
    }
    printf("\tOK\n");
}

int main() {
    char* image_file = "data/t10k-images-idx3-ubyte";
    char* labels_file = "data/t10k-labels-idx1-ubyte";
    printf("Chargement des paramètres\n");

    int* parameters = read_mnist_images_parameters(image_file);
    int nb_images = parameters[0];
    int height = parameters[1];
    int width = parameters[2];

    printf("OK\n");
    printf("Chargement des images\n");

    int*** images = read_mnist_images(image_file);

    printf("OK\n");
    printf("Chargement des labels\n");

    unsigned int* labels = read_mnist_labels(labels_file);

    printf("OK\n");
    printf("Vérification de l'accès en lecture\n");

    test_lecture(nb_images, width, height, images, labels);

    printf("OK\n");
    return 1;
}