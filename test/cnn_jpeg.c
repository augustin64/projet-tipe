#include <stdio.h>
#include <stdlib.h>

#include "../src/cnn/include/jpeg.h"
#include "../src/include/colors.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Pas de dataset en argument, test avorté\n");
        // On n'arrête pas le processus avce un code de sortie
        // pour pouvoir utiliser `make run-tests` dans des scripts
        // sans avoir à spécifier d'arguments supplémentaires
        return 0;
    }
    jpegDataset* dataset = loadJpegDataset(argv[1]);
    printf("Nombre de catégories: %d\n", dataset->numCategories);
    printf("Nombre d'images:      %d\n", dataset->numImages);
    printf("Taille des images:    %dx%d\n", dataset->width, dataset->height);
    #ifdef STORE_IMAGES_TO_RAM
    if (!dataset->images) {
        printf_error("Aucune image n'a été chargée\n");
        return 1;
    }
    #endif
    for (int i=0; i < (int)dataset->numImages; i++) {
        if (!dataset->fileNames[i]) {
            printf_error("Nom de fichier non chargé à l'index ");
            printf("%d\n", i);
            return 1;
        }
        #ifdef STORE_IMAGES_TO_RAM
        if (!dataset->images[i]) {
            printf_error("Image non chargée à l'index ");
            printf("%d\n", i);
            printf_error("Nom du fichier: ");
            printf("%s\n", dataset->fileNames[i]);
            return 1;
        }
        #endif
    }

    free_dataset(dataset);
    return 0;
}