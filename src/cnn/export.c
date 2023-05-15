#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>


#include "../common/include/colors.h"
#include "include/backpropagation.h"
#include "../common/include/mnist.h"
#include "include/neuron_io.h"
#include "include/struct.h"
#include "include/jpeg.h"
#include "include/free.h"
#include "include/cnn.h"


void help(char* call) {
    printf("Usage: %s ( print-poids-kernel-cnn | visual-propagation ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\tprint-poids-kernel-cnn\n");
    printf("\t\t--modele      | -m [FILENAME]\tFichier contenant le réseau entraîné\n");
    printf("\tvisual-propagation\n");
    printf("\t\t--modele      | -m [FILENAME]\tFichier contenant le réseau entraîné\n");
    printf("\t\t--out         | -o [BASE_FILENAME]\tLes images seront stockées dans ${out}_layer-${numéro de couche}_feature-${kernel_numero}.jpeg\n");
    printf("\t(mnist)\t--images      | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t(mnist)\t--numero      | -n [numero]\tNuméro de l'image dont la propagation veut être visualisée\n");
    printf("\t(jpeg)\t--jpeg-image  | -j [FILENAME]\tImage jpeg dont la propagation veut être visualisée.\n");
}


void print_poids_ker_cnn(char* modele) {
    Network* network = read_network(modele);
    int vus = 0;

    printf("{\n");
    for (int i=0; i < network->max_size-1; i++) {
        Kernel_cnn* kernel_cnn = network->kernel[i]->cnn;
        if (!(!kernel_cnn)) {
            if (vus != 0) {
                printf(",");
            }
            vus++;
            printf("\t\"%d\":[\n", i);
            for (int i=0; i < kernel_cnn->rows; i++) {
                printf("\t\t[\n");
                for (int j=0; j < kernel_cnn->columns; j++) {
                    printf("\t\t\t[\n");
                    for (int k=0; k < kernel_cnn->k_size; k++) {
                        printf("\t\t\t\t[");
                        for (int l=0; l < kernel_cnn->k_size; l++) {
                            printf("%lf", kernel_cnn->weights[i][j][k][l]);
                            if (l != kernel_cnn->k_size-1) {
                                printf(", ");
                            }
                        }
                        printf(" ]");
                        if (k != kernel_cnn->k_size-1) {
                            printf(",");
                        }
                        printf("\n");
                    }
                    printf("\t\t\t]");
                    if (j != kernel_cnn->columns-1) {
                        printf(",");
                    }
                    printf("\n");
                }
                printf("\t\t]");
                if (i != kernel_cnn->rows-1) {
                    printf(",");
                }
                printf("\n");
            }
            printf("\t]\n");
        }
    }
    printf("}\n");

    free_network(network);
}


void write_image(float** data, int width, int height, char* base_filename, int layer_id, int kernel_id) {
    int filename_length = strlen(base_filename) + (int)log10(layer_id+1)+1 + (int)log10(kernel_id+1)+1 + 21;
    char* filename = (char*)malloc(sizeof(char)*filename_length);

    sprintf(filename, "%s_layer-%d_feature-%d.jpeg", base_filename, layer_id, kernel_id);
    
    imgRawImage* image = (imgRawImage*)malloc(sizeof(imgRawImage));

    image->numComponents = 3;
    image->width = width;
    image->height = height;
    image->lpData = (unsigned char*)malloc(sizeof(unsigned char)*width*height*3);

    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            float color = fmax(fmin(data[i][j], 1.), 0.)*255;

            image->lpData[(i*width+j)*3] = color;
            image->lpData[(i*width+j)*3 + 1] = color;
            image->lpData[(i*width+j)*3 + 2] = color;
        }
    }

    storeJpegImageFile(image, filename);

    free(image->lpData);
    free(image);
    free(filename);
}


void visual_propagation(char* modele_file, char* mnist_images_file, char* out_base, int numero, char* jpeg_file) {
    Network* network = read_network(modele_file);

    if (mnist_images_file) {
        int* mnist_parameters = read_mnist_images_parameters(mnist_images_file);
        int*** images = read_mnist_images(mnist_images_file);

        int nb_elem = mnist_parameters[0];

        int width = mnist_parameters[1];
        int height = mnist_parameters[2];
        free(mnist_parameters);

        if (numero < 0 || numero >= nb_elem) {
            printf_error((char*)"Numéro d'image spécifié invalide.");
            printf(" Le fichier contient %d images.\n", nb_elem);
            exit(1);
        }

        // Write image to the network
        write_image_in_network_32(images[numero], height, width, network->input[0][0], false);

        // Free allocated memory from image reading
        for (int i=0; i < nb_elem; i++) {
            for (int j=0; j < width; j++) {
                free(images[i][j]);
            }
            free(images[i]);
        }
        free(images);
    } else {
        imgRawImage* image = loadJpegImageFile(jpeg_file);

        write_256_image_in_network(image->lpData, image->width, image->numComponents, network->width[0], network->input[0]);

        // Free allocated memory from image reading
        free(image->lpData);
        free(image);
    }
    forward_propagation(network);

    // Écriture des résultats
    for (int i=0; i < network->depth[0]; i++) {
        write_image(network->input[0][i], network->width[0], network->width[0], out_base, 0, i);
    }
    
    for (int i=1; i < network->size; i++) {
        if (!(!network->kernel[i-1]->nn)) {
            write_image(network->input[i][0], network->kernel[i-1]->nn->size_output, 1, out_base, i, 0);
        } else {
            for (int j=0; j < network->depth[i]; j++) {
                write_image(network->input[i][j], network->width[i], network->width[i], out_base, i, j);
            }
        }
    }

    free_network(network);
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf_error("Pas d'action spécifiée\n");
        help(argv[0]);
        return 1;
    }
    if (! strcmp(argv[1], "print-poids-kernel-cnn")) {
        char* modele = NULL; // Fichier contenant le modèle
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
                i += 2;
            } else {
                printf_warning("Option choisie inconnue: ");
                printf("%s\n", argv[i]);
                i++;
            }
        }
        if (!modele) {
            printf_error("Pas de modèle à utiliser spécifié.\n");
            return 1;
        }
        print_poids_ker_cnn(modele);
        return 0;
    }
    if (! strcmp(argv[1], "visual-propagation")) {
        char* modele = NULL; // Fichier contenant le modèle
        char* images = NULL; // Dossier contenant les images (mnist)
        char* out_base = NULL; // Préfixe du nom de fichier de sortie
        char* jpeg_image = NULL; // Image à regarder (jpeg)
        int numero = -1; // Numéro de l'image dans le dataset (mnist)
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--images"))||(! strcmp(argv[i], "-i"))) {
                if (images) {
                    printf_warning("Arguments conflictuels. L'image de type jpeg sera favorisée.\n");
                } else {
                    images = argv[i+1];
                }
                i += 2;
            } else if ((! strcmp(argv[i], "--jpeg-image"))||(! strcmp(argv[i], "-j"))) {
                if (images) {
                    printf_warning("Arguments conflictuels. L'image de type MNIST sera favorisée.\n");
                } else {
                    jpeg_image = argv[i+1];
                }
                i += 2;
            } else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out_base = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--numero"))||(! strcmp(argv[i], "-n"))) {
                numero = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else {
                printf_warning("Option choisie inconnue: ");
                printf("%s\n", argv[i]);
                i++;
            }
        }
        if (!modele) {
            printf_error("Pas de modèle à utiliser spécifié.\n");
            return 1;
        }
        if (!images && !jpeg_image) {
            printf_error("Pas de fichier d'images spécifié.\n");
            return 1;
        }
        if (!out_base) {
            printf_error("Pas de fichier de sortie spécifié.\n");
            return 1;
        }
        if (images) {
            if (numero == -1) {
                printf_error("Pas de numéro d'image spécifié.\n");
                return 1;
            }
            visual_propagation(modele, images, out_base, numero, NULL);
            return 0;
        }
        visual_propagation(modele, NULL, out_base, 0, jpeg_image);
        return 0;
    }
    printf_error("Option choisie non reconnue: ");
    printf("%s\n", argv[1]);
    help(argv[0]);
    return 1;
}