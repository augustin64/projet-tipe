#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


int** read_image(unsigned int width, unsigned int height, FILE* ptr) {
    unsigned char buffer[width*height];
    int** image = (int**)malloc(sizeof(int*)*height);
    
    size_t line_size = sizeof(int) * width;

    (void) !fread(buffer, sizeof(buffer), 1, ptr);

    for (int i=0; i < (int)height; i++) {
        int* line = (int*)malloc(line_size);
        for (int j=0; j < (int)width; j++) {
            line[j] = (int)buffer[j+i*width];
        }
        image[i] = line;
    }
    return image;
}


int* read_mnist_images_parameters(char* filename) {
    int* tab = (int*)malloc(sizeof(int)*3);
    FILE *ptr;
    
    ptr = fopen(filename, "rb");
    if (!ptr) {
        printf("Impossible de lire le fichier %s\n", filename);
        exit(1);
    }

    uint32_t magic_number;
    uint32_t number_of_images;
    unsigned int height;
    unsigned int width;

    (void) !fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    (void) !fread(&number_of_images, sizeof(uint32_t), 1, ptr);
    tab[0] = swap_endian(number_of_images);

    (void) !fread(&height, sizeof(unsigned int), 1, ptr);
    tab[1] = swap_endian(height);

    (void) !fread(&width, sizeof(unsigned int), 1, ptr);
    tab[2] = swap_endian(width);

    return tab;
}


uint32_t read_mnist_labels_nb_images(char* filename) {
    FILE *ptr;
    
    ptr = fopen(filename, "rb");
    if (!ptr) {
        printf("Impossible de lire le fichier %s\n", filename);
        exit(1);
    }

    uint32_t magic_number;
    uint32_t number_of_images;

    (void) !fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    (void) !fread(&number_of_images, sizeof(uint32_t), 1, ptr);
    number_of_images = swap_endian(number_of_images);

    return number_of_images;
}


int*** read_mnist_images(char* filename) {
    FILE *ptr;
    
    ptr = fopen(filename, "rb");
    if (!ptr) {
        printf("Impossible de lire le fichier %s\n", filename);
        exit(1);
    }

    uint32_t magic_number;
    uint32_t number_of_images;
    unsigned int height;
    unsigned int width;

    (void) !fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    (void) !fread(&number_of_images, sizeof(uint32_t), 1, ptr);
    number_of_images = swap_endian(number_of_images);

    (void) !fread(&height, sizeof(unsigned int), 1, ptr);
    height = swap_endian(height);

    (void) !fread(&width, sizeof(unsigned int), 1, ptr);
    width = swap_endian(width);

    int*** tab = (int***)malloc(sizeof(int**)*number_of_images);

    for (int i=0; i < (int)number_of_images; i++) {
        tab[i] = read_image(width, height, ptr);
    }
    return tab;
}

// Renvoie des labels formatés sous le format de la base MNIST
unsigned int* read_mnist_labels(char* filename) {
    FILE* ptr;

    ptr = fopen(filename, "rb");
    if (!ptr) {
        printf("Impossible de lire le fichier %s\n", filename);
        exit(1);
    }

    uint32_t magic_number;
    uint32_t number_of_items;

    (void) !fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    (void) !fread(&number_of_items, sizeof(uint32_t), 1, ptr);
    number_of_items = swap_endian(number_of_items);

    unsigned char buffer[number_of_items];
    (void) !fread(buffer, sizeof(unsigned char), number_of_items, ptr);

    unsigned int* labels = (unsigned int*)malloc(sizeof(unsigned int)*number_of_items);

    for (int i=0; i < (int)number_of_items; i++) {
        labels[i] = (unsigned int)buffer[i];
    }
    return labels;
}
