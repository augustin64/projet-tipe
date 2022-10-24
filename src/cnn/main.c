#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

#include "include/initialisation.h"
#include "../include/colors.h"
#include "include/function.h"
#include "include/creation.h"
#include "include/train.h"
#include "include/cnn.h"

#include "include/main.h"


void help(char* call) {
    printf("Usage: %s ( train | dev ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\tdev:\n");
    printf("\t\t--conv | -c\tTester la fonction dev_conv().\n");
    printf("\ttrain:\n");
    printf("\t\t--dataset | -d (mnist|jpg)\tFormat du set de données.\n");
    printf("\t(mnist)\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t(mnist)\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t (jpg) \t--datadir | -dd [FOLDER]\tDossier contenant les images.\n");
    printf("\t\t--epochs  | -e [int]\t\tNombre d'époques.\n");
    printf("\t\t--out     | -o [FILENAME]\tFichier où écrire le réseau de neurones.\n");
}


void dev_conv() {
    Network* network = create_network_lenet5(0, 0, TANH, GLOROT_NORMAL, 32, 1);
    forward_propagation(network);
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Pas d'action spécifiée\n");
        help(argv[0]);
        return 1;
    }
    if (! strcmp(argv[1], "dev")) {
        int option = 0;
        // 0 pour la fonction dev_conv()
        int i = 2;
        while (i < argc) {
            // Utiliser un switch serait sans doute plus élégant
            if ((! strcmp(argv[i], "--conv"))||(! strcmp(argv[i], "-c"))) {
                option = 0;
                i++;
            } else {
                printf("Option choisie inconnue: %s\n", argv[i]);
                i++;
            }
        }
        if (option == 0) {
            dev_conv();
            return 0;
        }
        printf("Option choisie inconnue: dev %d\n", option);
        return 1;
    }
    if (! strcmp(argv[1], "train")) {
        char* dataset = NULL;
        char* images_file = NULL;
        char* labels_file = NULL;
        char* data_dir = NULL;
        int epochs = EPOCHS;
        int dataset_type = 0;
        char* out = NULL;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--dataset"))||(! strcmp(argv[i], "-d"))) {
                dataset = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--images"))||(! strcmp(argv[i], "-i"))) {
                images_file = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--labels"))||(! strcmp(argv[i], "-l"))) {
                labels_file = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--datadir"))||(! strcmp(argv[i], "-dd"))) {
                data_dir = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--epochs"))||(! strcmp(argv[i], "-e"))) {
                epochs = strtol(argv[i+1], NULL, 10);
                i += 2;
            }
            else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            } else {
                printf("Option choisie inconnue: %s\n", argv[i]);
                i++;
            }
        }
        if ((dataset!=NULL) && !strcmp(dataset, "mnist")) {
            dataset_type = 0;
            if (!images_file) {
                printf("Pas de fichier d'images spécifié\n");
                return 1;
            }
            if (!labels_file) {
                printf("Pas de fichier de labels spécifié\n");
                return 1;
            }
        }
        else if ((dataset!=NULL) && !strcmp(dataset, "jpg")) {
            dataset_type = 1;
            if (!data_dir) {
                printf("Pas de dossier de données spécifié.\n");
                return 1;
            }
        }
        else {
            printf("Pas de type de dataset spécifié.\n");
            return 1;
        }
        if (!out) {
            printf("Pas de fichier de sortie spécifié, défaut: out.bin\n");
            out = "out.bin";
        }
        train(dataset_type, images_file, labels_file, data_dir, epochs, out);
        return 0;
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}