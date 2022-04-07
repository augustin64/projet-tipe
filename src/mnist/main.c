#include <stdlib.h>
#include <stdio.h>
#include <string.h>

void help(char* call) {
    printf("Usage: %s ( train | recognize ) [OPTIONS]\n\n", call);
    printf("OPTIONS:\n");
    printf("\ttrain:\n");
    printf("\t\t--batches | -b [int]\tNombre de batches\n");
    printf("\t\t--couches | -c [int]\tNombres de couches\n");
    printf("\t\t--neurons | -n [int]\tNombre de neurones sur la première couche\n");
    printf("\t\t--images  | -i [FILENAME]\tFichier contenant les images\n");
    printf("\t\t--labels  | -l [FILENAME]\tFichier contenant les labels\n");
    printf("\t\t--out     | -o [FILENAME]\tFichier où écrire le réseau de neurones\n");
    printf("\trecognize:\n");
    printf("\t\t--modele | -m [FILENAME]\tFichier contenant le réseau de neurones\n");
    printf("\t\t--in     | -i [FILENAME]\tFichier contenant les images à reconnaître\n");
    printf("\t\t--out    | -o (text|json)\tFormat de sortie\n");
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Pas d'action spécifiée\n");
        help(argv[0]);
        exit(1);
    }
    if (! strcmp(argv[1], "train")) {
        int batches = 5;
        int couches = 5;
        int neurons = 784;
        char* images = NULL;
        char* labels = NULL;
        char* out = NULL;
        int i=2;
        while (i < argc) {
            // Utiliser un switch serait sans doute plus élégant
            if ((! strcmp(argv[i], "--batches"))||(! strcmp(argv[i], "-b"))) {
                batches = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else
                if ((! strcmp(argv[i], "--couches"))||(! strcmp(argv[i], "-c"))) {
                couches = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else if ((! strcmp(argv[i], "--neurons"))||(! strcmp(argv[i], "-n"))) {
                neurons = strtol(argv[i+1], NULL, 10);
                i += 2;
            } else if ((! strcmp(argv[i], "--images"))||(! strcmp(argv[i], "-i"))) {
                images = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--labels"))||(! strcmp(argv[i], "-l"))) {
                labels = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! images) {
            printf("Pas de fichier d'images spécifié\n");
            exit(1);
        }
        if (! labels) {
            printf("Pas de fichier de labels spécifié\n");
            exit(1);
        }
        if (! out) {
            printf("Pas de fichier de sortie spécifié, default: out.bin\n");
            out = "out.bin";
        }
        // Entraînement en sourçant neural_network.c
        exit(0);
    }
    if (! strcmp(argv[1], "recognize")) {
        char* in = NULL;
        char* modele = NULL;
        char* out = NULL;
        int i=2;
        while(i < argc) {
            if ((! strcmp(argv[i], "--in"))||(! strcmp(argv[i], "-i"))) {
                in = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--modele"))||(! strcmp(argv[i], "-m"))) {
                modele = argv[i+1];
                i += 2;
            } else if ((! strcmp(argv[i], "--out"))||(! strcmp(argv[i], "-o"))) {
                out = argv[i+1];
                i += 2;
            } else {
                printf("%s : Argument non reconnu\n", argv[i]);
                i++;
            }
        }
        if (! in) {
            printf("Pas d'entrée spécifiée\n");
            exit(1);
        }
        if (! modele) {
            printf("Pas de modèle spécifié\n");
            exit(1);
        }
        if (! out) {
            out = "text";
        }
        // Reconnaissance puis affichage des données sous le format spécifié
        exit(0);
    }
    printf("Option choisie non reconnue: %s\n", argv[1]);
    help(argv[0]);
    return 1;
}