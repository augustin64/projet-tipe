#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "../include/memory_management.h"
#include "../mnist/include/mnist.h"
#include "include/neuron_io.h"
#include "include/struct.h"
#include "include/jpeg.h"
#include "include/free.h"
#include "include/cnn.h"


void test_network_mnist(Network* network, char* images_file, char* labels_file, bool preview_fails) {
    (void)preview_fails; // Inutilisé pour le moment
    int width, height; // Dimensions des images
    int nb_elem; // Nombre d'éléments
    int maxi; // Catégorie reconnue

    int accuracy = 0; // Nombre d'images reconnues

    // Load image
    int* mnist_parameters = read_mnist_images_parameters(images_file);

    int*** images = read_mnist_images(images_file);
    unsigned int* labels = read_mnist_labels(labels_file);

    nb_elem = mnist_parameters[0];

    width = mnist_parameters[1];
    height = mnist_parameters[2];
    free(mnist_parameters);

    // Load image in the first layer of the Network
    for (int i=0; i < nb_elem; i++) {
        if(i %(nb_elem/100) == 0) {
            printf("Avancement: %.0f%%\r",  100*i/(float)nb_elem);
            fflush(stdout);
        }
        write_image_in_network_32(images[i], height, width, network->input[0][0]);
        forward_propagation(network);
        maxi = indice_max(network->input[network->size-1][0][0], 10);

        if (maxi == (int)labels[i]) {
            accuracy++;
        }

        for (int j=0; j < height; j++) {
            free(images[i][j]);
        }
        free(images[i]);
    }
    free(images);
    printf("%d Images. Taux de réussite: %.2f%%\n", nb_elem, 100*accuracy/(float)nb_elem);
}


void test_network_jpg(Network* network, char* data_dir, bool preview_fails) {
    (void)preview_fails; // Inutilisé pour le moment
    jpegDataset* dataset = loadJpegDataset(data_dir);

    int accuracy = 0;
    int maxi;

    for (int i=0; i < (int)dataset->numImages; i++) {
        if(i %(dataset->numImages/100) == 0) {
            printf("Avancement: %.1f%%\r",  1000*i/(float)dataset->numImages);
            fflush(stdout);
        }
        write_image_in_network_260(dataset->images[i], dataset->height, dataset->height, network->input[0]);
        forward_propagation(network);
        maxi = indice_max(network->input[network->size-1][0][0], 50);

        if (maxi == (int)dataset->labels[i]) {
            accuracy++;
        }

        free(dataset->images[i]);
    }

    printf("%d Images. Taux de réussite: %.2f%%\n", dataset->numImages, 100*accuracy/(float)dataset->numImages);
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}


void test_network(int dataset_type, char* modele, char* images_file, char* labels_file, char* data_dir, bool preview_fails) {
    Network* network = read_network(modele);

    if (dataset_type == 0) {
        test_network_mnist(network, images_file, labels_file, preview_fails);
    } else {
        test_network_jpg(network, data_dir, preview_fails);
    }

    free_network(network);
}


void recognize_mnist(Network* network, char* input_file, char* out) {
    int width, height; // Dimensions de l'image
    int nb_elem; // Nombre d'éléments

    // Load image
    int* mnist_parameters = read_mnist_images_parameters(input_file);
    int*** images = read_mnist_images(input_file);
    nb_elem = mnist_parameters[0];

    width = mnist_parameters[1];
    height = mnist_parameters[2];
    free(mnist_parameters);

    if (! strcmp(out, "json")) {
        printf("{\n");
    } else {
        printf("Image\tCatégorie détectée\n");
    }
    // Load image in the first layer of the Network
    for (int i=0; i < nb_elem; i++) {
        if (! strcmp(out, "json")) {
            printf("\"%d\" : [", i);
        }

        write_image_in_network_32(images[i], height, width, network->input[0][0]);
        forward_propagation(network);

    
        if (! strcmp(out, "json")) {
            for (int j=0; j < 10; j++) {
                printf("%f", network->input[network->size-1][0][0][j]);

                if (j+1 < 10) {
                    printf(", ");
                }
            }
        } else {
            printf("%d\t%d\n", i, indice_max(network->input[network->size-1][0][0], 10));
        }

        if (! strcmp(out, "json")) {
            if (i+1 < nb_elem) {
                printf("],\n");
            } else {
                printf("]\n");
            }
        }

        for (int j=0; j < height; j++) {
            free(images[i][j]);
        }
        free(images[i]);
    }
    if (! strcmp(out, "json")) {
        printf("}\n");
    }

    free(images);
}

void recognize_jpg(Network* network, char* input_file, char* out) {
    int width, height; // Dimensions de l'image
    int maxi;

    imgRawImage* image = loadJpegImageFile(input_file);
    width = image->width;
    height = image->height;

    if (! strcmp(out, "json")) {
        printf("{\n");
        printf("\"0\" : [");
    }

    // Load image in the first layer of the Network
    write_image_in_network_260(image->lpData, height, width, network->input[0]);
    forward_propagation(network);


    if (! strcmp(out, "json")) {
        for (int j=0; j < 50; j++) {
            printf("%f", network->input[network->size-1][0][0][j]);

            if (j+1 < 10) {
                printf(", ");
            }
        }
    } else {
        maxi = indice_max(network->input[network->size-1][0][0], 50);
        printf("Catégorie reconnue: %d\n", maxi);
    }

    if (! strcmp(out, "json")) {
        printf("]\n");
        printf("}\n");
    }

    free(image->lpData);
    free(image);
}

void recognize(int dataset_type, char* modele, char* input_file, char* out) {
    Network* network = read_network(modele);

    if (dataset_type == 0) {
        recognize_mnist(network, input_file, out);
    } else {
        recognize_jpg(network, input_file, out);
    }

    free_network(network);
}