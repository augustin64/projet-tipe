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
    printf("Usage: %s ( print-poids | print-biais | creer-reseau ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\tprint-poids:\n");
    printf("\t\t--reseau | -r [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\tprint-biais:\n");
    printf("\t\t--reseau | -r [FILENAME]\tFichier contenant le réseau de neurones.\n");
    printf("\tcount-labels:\n");
    printf("\t\t--labels | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\tcreer-reseau:\n");
    printf("\t\t--out    | -o [FILENAME]\tFichier où écrire le réseau de neurones.\n");
    printf("\t\t--number | -n [int]\tNuméro à privilégier\n");
}


void print_biais(char* filename) {
    Reseau* reseau = lire_reseau(".cache/reseau.bin");

    for (int i=1; i < reseau->nb_couches -1; i++) {
        printf("Couche %d\n", i);
        for (int j=0; j < reseau->couches[i]->nb_neurones; j++) {
            printf("Couche %d\tNeurone %d\tBiais: %f\n", i, j, reseau->couches[i]->neurones[j]->biais);
        }
    }
    suppression_du_reseau_neuronal(reseau);
}

void print_poids(char* filename) {
    Reseau* reseau = lire_reseau(".cache/reseau.bin");

    for (int i=0; i < reseau->nb_couches -1; i++) {
        printf("Couche %d\n", i);
        for (int j=0; j < reseau->couches[i]->nb_neurones; j++) {
            printf("Couche %d\tNeurone %d\tPoids: [", i, j);
            for (int k=0; k < reseau->couches[i+1]->nb_neurones; k++) {
                printf("%f, ", reseau->couches[i]->neurones[j]->poids_sortants[k]);
            }
            printf("]\n");
        }
    }
    suppression_du_reseau_neuronal(reseau);
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

void creer_reseau(char* filename, int sortie) {
    Reseau* reseau = malloc(sizeof(Reseau));
    Couche* couche;
    Neurone* neurone;
    reseau->nb_couches = 3;
    
    reseau->couches = malloc(sizeof(Couche*)*reseau->nb_couches);
    int neurones_par_couche[4] = {784, 1, 10, 0};
    for (int i=0; i < reseau->nb_couches; i++) {
        reseau->couches[i] = malloc(sizeof(Couche));
        couche = reseau->couches[i];
        couche->nb_neurones = neurones_par_couche[i];
        couche->neurones = malloc(sizeof(Neurone*)*couche->nb_neurones);
        for (int j=0; j < couche->nb_neurones; j++) {
            couche->neurones[j] = malloc(sizeof(Neurone));
            neurone = couche->neurones[j];

            neurone->activation = 0.;
            neurone->biais = 0.;
            neurone->z = 0.;

            neurone->d_activation = 0.;
            neurone->d_biais = 0.;
            neurone->d_z = 0.;

            neurone->poids_sortants = malloc(sizeof(float)*neurones_par_couche[i+1]);
            neurone->d_poids_sortants = malloc(sizeof(float)*neurones_par_couche[i+1]);
            for (int k=0; k < neurones_par_couche[i+1]; k++) {
                neurone->poids_sortants[k] = 0.;
                neurone->d_poids_sortants[k] = 0.;
            }
        }
    }

    for (int j=0; j < neurones_par_couche[0]; j++) {
        reseau->couches[0]->neurones[j]->poids_sortants[0] = 1;
    }
    reseau->couches[1]->neurones[0]->poids_sortants[sortie] = 1;
    ecrire_reseau(filename, reseau);
    suppression_du_reseau_neuronal(reseau);
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
        print_poids(filename);
        exit(1);
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
        print_biais(filename);
        exit(1);
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