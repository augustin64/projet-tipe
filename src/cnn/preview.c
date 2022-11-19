#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include "include/jpeg.h"


void print_image(unsigned char* image, int height, int width) {

    for (int i=0; i < (int)width; i++) {
        for (int j=0; j < (int)height; j++) {
            printf("\x1b[38;2;%d;%d;%dm#\x1b[0m", image[((i*width)+j)*3 + 0], image[((i*width)+j)*3 + 1], image[((i*width)+j)*3 + 2]);
        }
        printf("\n");
    }
}

void preview_images(char* path, int limit) {
    jpegDataset* dataset = loadJpegDataset(path);
    imgRawImage* image;

    if (limit == -1) {
        limit = dataset->numImages;
    }
    for (int i=0; i < limit; i++) {
        printf("--- Image %d : %d ---\n", i, dataset->labels[i]);

        if (!dataset->images[i]) {
            image = loadJpegImageFile(dataset->fileNames[i]);
            dataset->images[i] = image->lpData;
            free(image);
        }
        print_image(dataset->images[i], dataset->height, dataset->width);

        free(dataset->images[i]);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Utilisation: %s [DIRECTORY] (opt:nombre d'images)\n", argv[0]);
        return 1;
    }
    int limit = -1;
    if (argc > 2) {
        limit = strtol(argv[2], NULL, 10);
    }
    preview_images(argv[1], limit);
    return 0;
}
