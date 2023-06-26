#include <stdlib.h>
#include <stdio.h>
#include <time.h>


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

    // Calcul du temps de chargement des images une à une
    clock_t start_time, end_time;

    int N = min(100000, dataset->numImages);
    start_time = clock();
    printf("Chargement de %d images\n", N);
    for (int i=0; i < N; i++) {
        imgRawImage* image = loadJpegImageFile(dataset->fileNames[i]);
        free(image->lpData);
        free(image);
    }
    printf("OK\n");
    end_time = clock();
    printf("Temps par image (calculé sur une moyenne de %d): ", N);
    printf_time((end_time - start_time)/N);
    printf("\n");

    for (int i=0; i < (int)dataset->numImages; i++) {
        if (!dataset->fileNames[i]) {
            printf_error("Nom de fichier non chargé à l'index ");
            printf("%d\n", i);
            return 1;
        }
    }

    free_dataset(dataset);
    return 0;
}