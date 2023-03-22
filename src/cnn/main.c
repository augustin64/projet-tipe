#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <float.h>

#include "include/initialisation.h"
#include "include/test_network.h"
#include "../include/colors.h"
#include "include/function.h"
#include "include/creation.h"
#include "include/train.h"
#include "include/cnn.h"

#include "include/main.h"


void help(char* call) {
    printf("Usage: %s ( train | recognize | test ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\ttrain:\n");
    printf("\t\t--dataset | -d (mnist|jpg)\tFormat du set de données.\n");
    printf("\t(mnist)\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t(mnist)\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t (jpg) \t--datadir | -dd [FOLDER]\tDossier contenant les images.\n");
    printf("\t\t--recover | -r [FILENAME]\tRécupérer depuis un modèle existant.\n");
    printf("\t\t--epochs  | -e [int]\t\tNombre d'époques.\n");
    printf("\t\t--out     | -o [FILENAME]\tFichier où écrire le réseau de neurones.\n");
    printf("\trecognize:\n");
    printf("\t\t--dataset | -d (mnist|jpg)\tFormat de l'image à reconnaître.\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau entraîné.\n");
    printf("\t\t--input   | -i [FILENAME]\tImage jpeg ou fichier binaire à reconnaître.\n");
    printf("\t\t--out     | -o (text|json)\tFormat de sortie.\n");
    printf("\ttest:\n");
    printf("\t\t--modele  | -m [FILENAME]\tFichier contenant le réseau entraîné.\n");
    printf("\t\t--dataset | -d (mnist|jpg)\tFormat du set de données.\n");
    printf("\t(mnist)\t--images  | -i [FILENAME]\tFichier contenant les images.\n");
    printf("\t(mnist)\t--labels  | -l [FILENAME]\tFichier contenant les labels.\n");
    printf("\t (jpg) \t--datadir | -dd [FOLDER]\tDossier contenant les images.\n");
    printf("\t\t--preview-fails | -p\t\tAfficher les images ayant échoué.\n");
}



int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf_error("Pas d'action spécifiée\n");
        help(argv[0]);
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
        char* recover = NULL;
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
            } else if ((! strcmp(argv[i], "--recover"))||(! strcmp(argv[i], "-r"))) {
                recover = argv[i+1];
                i += 2;
            } else {
                printf_warning("Option choisie inconnue: ");
                printf("%s\n", argv[i]);
                i++;
            }
        }
        if ((dataset!=NULL) && !strcmp(dataset, "mnist")) {
            dataset_type = 0;
            if (!images_file) {
                printf_error("Pas de fichier d'images spécifié\n");
                return 1;
            }
            if (!labels_file) {
                printf_error("Pas de fichier de labels spécifié\n");
                return 1;
            }
        }
        else if ((dataset!=NULL) && !strcmp(dataset, "jpg")) {
            dataset_type = 1;
            if (!data_dir) {
                printf_error("Pas de dossier de données spécifié.\n");
                return 1;
            }
        }
        else {
            printf_error("Pas de type de dataset spécifié.\n");
            return 1;
        }
        if (!out) {
            printf("Pas de fichier de sortie spécifié, défaut: out.bin\n");
            out = "out.bin";
        }
        train(dataset_type, images_file, labels_file, data_dir, epochs, out, recover);
        return 0;
    }
    if (! strcmp(argv[1], "test")) {
        char* dataset = NULL; // mnist ou jpg
        char* modele = NULL; // Fichier contenant le modèle
        char* images_file = NULL; // Fichier d'images (mnist)
        char* labels_file = NULL; // Fichier de labels (mnist)
        char* data_dir = NULL; // Dossier d'images (jpg)
        int dataset_type; // Type de dataset (0 pour mnist, 1 pour jpg)
        bool preview_fails = false;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--dataset"))||(! strcmp(argv[i], "-d"))) {
                dataset = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
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
            else if ((! strcmp(argv[i], "--preview-fails"))||(! strcmp(argv[i], "-p"))) {
                preview_fails = true;
                i++;
            }
            else {
                printf_warning("Option choisie inconnue: ");
                printf("%s\n", argv[i]);
                i++;
            }
        }
        if ((dataset!=NULL) && !strcmp(dataset, "mnist")) {
            dataset_type = 0;
            if (!images_file) {
                printf_error("Pas de fichier d'images spécifié\n");
                return 1;
            }
            if (!labels_file) {
                printf_error("Pas de fichier de labels spécifié\n");
                return 1;
            }
        }
        else if ((dataset!=NULL) && !strcmp(dataset, "jpg")) {
            dataset_type = 1;
            if (!data_dir) {
                printf_error("Pas de dossier de données spécifié.\n");
                return 1;
            }
        }
        else {
            printf_error("Pas de type de dataset spécifié.\n");
            return 1;
        }

        if (!modele) {
            printf_error("Pas de modèle à utiliser spécifié.\n");
            return 1;
        }
        (void)test_network(dataset_type, modele, images_file, labels_file, data_dir, preview_fails, true, false);
        return 0;
    }
    if (! strcmp(argv[1], "recognize")) {
        char* dataset = NULL; // mnist ou jpg
        char* modele = NULL; // Fichier contenant le modèle
        char* input_file = NULL; // Image à reconnaître
        char* out = NULL;
        int dataset_type;
        int i = 2;
        while (i < argc) {
            if ((! strcmp(argv[i], "--dataset"))||(! strcmp(argv[i], "-d"))) {
                dataset = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            }
            else if ((! strcmp(argv[i], "--input"))||(! strcmp(argv[i], "-i"))) {
                input_file = argv[i+1];
                i += 2;
            } else {
                printf_warning("Option choisie inconnue: ");
                printf("%s\n", argv[i]);
                i++;
            }
        }
        if ((dataset!=NULL) && !strcmp(dataset, "mnist")) {
            dataset_type = 0;
        } else if ((dataset!=NULL) && !strcmp(dataset, "jpg")) {
            dataset_type = 1;
        }
        else {
            printf_error("Pas de type de dataset spécifié.\n");
            return 1;
        }
        if (!input_file) {
            printf_error("Pas de fichier d'entrée spécifié, rien à faire.\n");
            return 1;
        }
        if (!out) {
            out = "text";
        }
        if (!modele) {
            printf_error("Pas de modèle à utiliser spécifié.\n");
            return 1;
        }
        recognize(dataset_type, modele, input_file, out);
        return 0;
    }
    printf_error("Option choisie non reconnue: ");
    printf("%s\n", argv[1]);
    help(argv[0]);
    return 1;
}