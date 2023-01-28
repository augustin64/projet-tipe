#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include "../include/utils.h"

#include "include/jpeg.h"


void print_image(unsigned char* image, int height, int width) {
    int red, green, blue;
    for (int i=0; i < (int)height/2; i++) {
        for (int j=0; j < (int)width; j++) {
            red = (image[((2*i*width)+j)*3 + 0] + image[(((2*i+1)*width)+j)*3 + 0])/2;
            green = (image[((2*i*width)+j)*3 + 1] + image[(((2*i+1)*width)+j)*3 + 1])/2;;
            blue = (image[((2*i*width)+j)*3 + 2] + image[(((2*i+1)*width)+j)*3 + 2])/2;;

            // Make the text color opposed to background color
            printf("\x1b[38;2;%d;%d;%dm", 255-red, 255-green, 255-blue);

            printf("\x1b[48;2;%d;%d;%dm ", red, green, blue);
        }
        printf("\x1b[0m\n");
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
            gree(image);
        }
        print_image(dataset->images[i], dataset->height, dataset->width);

        gree(dataset->images[i]);
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
