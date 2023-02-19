#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#include "include/free.h"
#include "include/struct.h"
#include "include/neuron_io.h"


void help(char* call) {
    printf("Usage: %s ( print-poids-kernel-cnn ) [OPTIONS]\n\n", call);
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



int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Pas d'action spécifiée\n");
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
                printf("Option choisie inconnue: %s\n", argv[i]);
                i++;
            }
        }
        if (!modele) {
            printf("Pas de modèle à utiliser spécifié.\n");
            return 1;
        }
        print_poids_ker_cnn(modele);
        return 0;
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}