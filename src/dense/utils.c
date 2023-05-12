#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "../common/include/mnist.h"
#include "include/neural_network.h"
#include "include/neuron_io.h"

/*
Contient un ensemble de fonctions utiles pour le débogage
*/
void help(char* call) {
    printf("Usage: %s ( print-poids | print-biais | creer-reseau | patch-network ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\tprint-poids:\n");
    printf("\t\t--reseau | -r [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\tprint-biais:\n");
    printf("\t\t--reseau | -r [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\tcount-labels:\n");
    printf("\t\t--labels | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\tcreer-reseau:\n");
    printf("\t\t--out    | -o [FILENAME]\tFichier où écrire le réseau de neurones.\n");
    printf("\t\t--number | -n [int]\tNuméro à privilégier.\n");
    printf("\tpatch-network:\n");
    printf("\t\t--network | -n [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\t\t--delta   | -d [FILENAME]\tFichier de patch à utiliser.\n");
    printf("\tprint-images:\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\tprint-poids-neurone:\n");
    printf("\t\t--reseau | -r [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\t\t--neurone | -n [int]\tNuméro du neurone dont il faut afficher les poids.\n");
}


void print_bias(char* filename) {
    Network* network = read_network(filename);

    for (int i=1; i < network->nb_layers -1; i++) {
        printf("Couche %d\n", i);
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {
            printf("Couche %d\tNeurone %d\tBiais: %f\n", i, j, network->layers[i]->neurons[j]->bias);
        }
    }
    deletion_of_network(network);
}

void print_weights(char* filename) {
    Network* network = read_network(filename);

    for (int i=0; i < network->nb_layers -1; i++) {
        printf("Couche %d\n", i);
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {
            printf("Couche %d\tNeurone %d\tPoids: [", i, j);
            for (int k=0; k < network->layers[i+1]->nb_neurons; k++) {
                printf("%f, ", network->layers[i]->neurons[j]->weights[k]);
            }
            printf("]\n");
        }
    }
    deletion_of_network(network);
}

void count_labels(char* filename) {
    uint32_t number_of_images = read_mnist_labels_nb_images(filename);
    unsigned int* labels = read_mnist_labels(filename);

    unsigned int tab[10];

    for (int i=0; i < 10; i++) {
        tab[i] = 0;
    }

    for (int i=0; i < (int)number_of_images; i++) {
        tab[(int)labels[i]]++;
    }

    for (int i=0; i < 10; i++) {
        printf("Nombre de %d: %u\n", i, tab[i]);
    }
    free(labels);
}

void create_network(char* filename, int sortie) {
    Network* network = (Network*)malloc(sizeof(Network));
    Layer* layer;
    Neuron* neuron;
    network->nb_layers = 3;
    
    network->layers = (Layer**)malloc(sizeof(Layer*)*network->nb_layers);
    int neurons_per_layer[4] = {784, 1, 10, 0};
    for (int i=0; i < network->nb_layers; i++) {
        layer = (Layer*)malloc(sizeof(Layer));
        layer->nb_neurons = neurons_per_layer[i];
        layer->neurons = (Neuron**)malloc(sizeof(Neuron*)*layer->nb_neurons);
        for (int j=0; j < layer->nb_neurons; j++) {
            neuron = (Neuron*)malloc(sizeof(Neuron));

            neuron->bias = 0.;
            neuron->z = 0.;

            neuron->back_bias = 0.;
            neuron->last_back_bias = 0.;
            if (i != network->nb_layers-1) {
                neuron->weights = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
                neuron->back_weights = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
                neuron->last_back_weights = (float*)malloc(sizeof(float)*neurons_per_layer[i+1]);
                for (int k=0; k < neurons_per_layer[i+1]; k++) {
                    neuron->weights[k] = 0.;
                    neuron->back_weights[k] = 0.;
                    neuron->last_back_weights[k] = 0.;
                }
            }
            layer->neurons[j] = neuron;
        }
        network->layers[i] = layer;
    }
    for (int j=0; j < neurons_per_layer[0]; j++) {
        network->layers[0]->neurons[j]->weights[0] = 1;
    }
    network->layers[1]->neurons[0]->weights[sortie] = 1;
    write_network(filename, network);
    deletion_of_network(network);
}


void patch_stored_network(char* network_filename, char* delta_filename) {
    // Apply patch to a network stored in a file
    Network* network = read_network(network_filename);
    Network* delta = read_delta_network(delta_filename);

    patch_network(network, delta, 1);

    write_network(network_filename, network);
    deletion_of_network(network);
    deletion_of_network(delta);
}


void print_images(char* filename) {
    int* parameters = read_mnist_images_parameters(filename);

    int nb_elem = parameters[0];
    int width = parameters[1];
    int height = parameters[2];
    free(parameters);

    int*** images = read_mnist_images(filename);

    printf("[\n");
    for (int i=0; i < nb_elem; i++) {
        printf("\t[\n");
        for (int j=0; j < height; j++) {
            printf("\t\t[");
            for (int k=0; k < width; k++) {
                if (k != width -1)
                    printf("%d, ", images[i][j][k]);
                else
                    printf("%d", images[i][j][k]);
            }
            if (j != height -1)
                printf("],\n");
            else
                printf("]\n");
            free(images[i][j]);
        }
        if (i != nb_elem -1)
            printf("\t],\n");
        else
            printf("\t]\n");
        free(images[i]);
    }
    free(images);
    printf("]\n");
}


void print_poids_neurone(char* filename, int num_neurone) {
    Network* network = read_network(filename);
    int nb_layers = network->nb_layers;

    Layer* layer = network->layers[nb_layers-2];
    int nb_neurons = layer->nb_neurons;
    printf("[\n");
    for (int i=0; i < nb_neurons; i++) {
        printf("%f", layer->neurons[i]->weights[num_neurone]);
        if (i != nb_neurons -1)
            printf(", ");
        else
            printf("\n");
    }
    printf("]\n");
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Pas d'action spécifiée\n");
        help(argv[0]);
        exit(1);
    }
    if (! strcmp(argv[1], "print-poids")) {
        char* filename = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--reseau"))||(! strcmp(argv[i], "-r"))) {
                filename = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! filename) {
            printf("Pas de fichier spécifié, utilisation de '.cache/reseau.bin'\n");
            filename = ".cache/reseau.bin";
        }
        print_weights(filename);
        exit(0);
    } else if (! strcmp(argv[1], "print-biais")) {
        char* filename = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--reseau"))||(! strcmp(argv[i], "-r"))) {
                filename = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! filename) {
            printf("Pas de fichier spécifié, utilisation de '.cache/reseau.bin'\n");
            filename = ".cache/reseau.bin";
        }
        print_bias(filename);
        exit(0);
    } else if (! strcmp(argv[1], "creer-reseau")) {
        char* out = NULL;
        int n = -1;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--number"))||(! strcmp(argv[i], "-n"))) {
                n = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        create_network(out, n);
        exit(0);
    } else if (! strcmp(argv[1], "count-labels")) {
        char* labels = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--labels"))||(! strcmp(argv[i], "-l"))) {
                labels = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! labels) {
            printf("Pas de fichier spécifié, défaut: 'data/mnist/train-labels-idx1-ubyte'\n");
            labels = "data/mnist/train-labels-idx1-ubyte";
        }
        count_labels(labels);
        exit(0);
    } else if (! strcmp(argv[1], "patch-network")) {
        char* network = NULL;
        char* delta = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--network"))||(! strcmp(argv[i], "-n"))) {
                network = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--delta"))||(! strcmp(argv[i], "-d"))) {
                delta = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (!network) {
            printf("--network: Argument obligatoire.\n");
            exit(1);
        }
        if (!delta) {
            printf("--delta: Argument obligatoire.\n");
            exit(1);
        }
        patch_stored_network(network, delta);
        exit(0);
    } else if (! strcmp(argv[1], "print-images")) {
        char* images = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--images"))||(! strcmp(argv[i], "-i"))) {
                images = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (!images) {
            printf("--images: Argument obligatoire.\n");
            exit(1);
        }
        print_images(images);
        exit(0);
    } else if (! strcmp(argv[1], "print-poids-neurone")) {
        char* reseau = NULL;
        int neurone = 0;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--reseau"))||(! strcmp(argv[i], "-r"))) {
                reseau = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--neurone"))||(! strcmp(argv[i], "-n"))) {
                neurone = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (!reseau) {
            printf("--reseau: Argument obligatoire.\n");
            exit(1);
        }
        print_poids_neurone(reseau, neurone);
        exit(0);
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}