#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <float.h>

#include "neural_network.c"
#include "neuron_io.c"
#include "mnist.c"

/*
Contient un ensemble de fonctions utiles pour le débogage
*/
void help(char* call) {
    printf("Usage: %s ( print-poids | print-bias | creer-network ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\tprint-poids:\n");
    printf("\t\t--network | -r [FILENAME]\tFichier contenant le réseau de neurons.\n");
    printf("\tprint-bias:\n");
    printf("\t\t--network | -r [FILENAME]\tFichier contenant le réseau de neurons.\n");
    printf("\tcount-labels:\n");
    printf("\t\t--labels | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\tcreer-network:\n");
    printf("\t\t--out    | -o [FILENAME]\tFichier où écrire le réseau de neurons.\n");
    printf("\t\t--number | -n [int]\tNuméro à privilégier\n");
}


void print_bias(char* filename) {
    Network* network = read_network(".cache/network.bin");

    for (int i=1; i < network->nb_layers -1; i++) {
        printf("Layer %d\n", i);
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {
            printf("Layer %d\tNeuron %d\tBiais: %f\n", i, j, network->layers[i]->neurons[j]->bias);
        }
    }
    deletion_of_network(network);
}

void print_poids(char* filename) {
    Network* network = read_network(".cache/network.bin");

    for (int i=0; i < network->nb_layers -1; i++) {
        printf("Layer %d\n", i);
        for (int j=0; j < network->layers[i]->nb_neurons; j++) {
            printf("Layer %d\tNeuron %d\tPoids: [", i, j);
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

    unsigned int* labels = malloc(sizeof(unsigned int)*number_of_images);
    labels = read_mnist_labels(filename);

    unsigned int* tab[10];

    for (int i=0; i < 10; i++) {
        tab[i] = 0;
    }

    for (int i=0; i < number_of_images; i++) {
        tab[(int)labels[i]]++;
    }

    for (int i=0; i < 10; i++) {
        printf("Nombre de %d: %d\n", i, tab[i]);
    }
}

void create_network(char* filename, int sortie) {
    Network* network = malloc(sizeof(Network));
    Layer* layer;
    Neuron* neuron;
    network->nb_layers = 3;
    
    network->layers = malloc(sizeof(Layer*)*network->nb_layers);
    int neurons_per_layer[4] = {784, 1, 10, 0};
    for (int i=0; i < network->nb_layers; i++) {
        network->layers[i] = malloc(sizeof(Layer));
        layer = network->layers[i];
        layer->nb_neurons = neurons_per_layer[i];
        layer->neurons = malloc(sizeof(Neuron*)*layer->nb_neurons);
        for (int j=0; j < layer->nb_neurons; j++) {
            layer->neurons[j] = malloc(sizeof(Neuron));
            neuron = layer->neurons[j];

            neuron->bias = 0.;
            neuron->z = 0.;

            neuron->back_bias = 0.;
            neuron->last_back_bias = 0.;

            neuron->weights = malloc(sizeof(float)*neurons_per_layer[i+1]);
            neuron->back_weights = malloc(sizeof(float)*neurons_per_layer[i+1]);
            neuron->last_back_weights = malloc(sizeof(float)*neurons_per_layer[i+1]);
            for (int k=0; k < neurons_per_layer[i+1]; k++) {
                neuron->weights[k] = 0.;
                neuron->back_weights[k] = 0.;
                neuron->last_back_weights[k] = 0.;
            }
        }
    }

    for (int j=0; j < neurons_per_layer[0]; j++) {
        network->layers[0]->neurons[j]->weights[0] = 1;
    }
    network->layers[1]->neurons[0]->weights[sortie] = 1;
    write_network(filename, network);
    deletion_of_network(network);
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
            if ((! strcmp(argv[i], "--network"))||(! strcmp(argv[i], "-r"))) {
                filename = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! filename) {
            printf("Pas de fichier spécifié, utilisation de '.cache/network.bin'\n");
            filename = ".cache/network.bin";
        }
        print_poids(filename);
        exit(1);
    } else if (! strcmp(argv[1], "print-bias")) {
        char* filename = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--network"))||(! strcmp(argv[i], "-r"))) {
                filename = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! filename) {
            printf("Pas de fichier spécifié, utilisation de '.cache/network.bin'\n");
            filename = ".cache/network.bin";
        }
        print_bias(filename);
        exit(1);
    } else if (! strcmp(argv[1], "creer-network")) {
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
        exit(1);
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}