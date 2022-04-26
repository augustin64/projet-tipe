#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#include "mnist.c"


// Prévisualise un chiffre écrit à la main
// de taille width x height
void print_image(unsigned int width, unsigned int height, int** image) {
    char tab[] = {' ', '.', ':', '%', '#', '\0'};

    for (int i=0; i < height; i++) {
        for (int j=0; j < width; j++) {
            printf("%c", tab[image[i][j]/52]);
        }
        printf("\n");
    }
}

void preview_images(char* images_file, char* labels_file) {
    int* parameters = read_mnist_images_parameters(images_file);
    
    int number_of_images = parameters[0];
    int width = parameters[1];
    int height = parameters[2];
    
    unsigned int* labels = read_mnist_labels(labels_file);
    int*** images = read_mnist_images(images_file);

    for (int i=0; i < number_of_images; i++) {
        printf("--- Number %d : %d ---\n", i, labels[i]);
        print_image(width, height, images[i]);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Utilisation: %s [IMAGES FILE] [LABELS FILE]\n", argv[0]);
        return 1;
    }
    preview_images(argv[1], argv[2]);
    return 0;
}
