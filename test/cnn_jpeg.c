#include <stdlib.h>
#include <stdio.h>
#include <omp.h>


#include "../src/common/include/colors.h"
#include "../src/common/include/utils.h"
#include "../src/cnn/include/jpeg.h"

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Pas de dataset en argument, test avorté\n");
        // On arrête le processus avec un code de sortie 0
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

    // Calcul du temps de chargement des images une à une
    double start_time, end_time;

    int N = min(100000, dataset->numImages);
    start_time = omp_get_wtime();
    printf("Chargement de %d images\n", N);
    for (int i=0; i < N; i++) {
        imgRawImage* image = loadJpegImageFile(dataset->fileNames[i]);
        free(image->lpData);
        free(image);
    }
    printf("OK\n");
    end_time = omp_get_wtime();
    printf("Temps par image (calculé sur une moyenne de %d): %lf s\n", N, (end_time - start_time)/N);

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