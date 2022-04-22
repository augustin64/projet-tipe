#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>


uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


// Renvoie une image sous forme d'un int**
int** read_image(unsigned int width, unsigned int height, FILE* ptr) {
    unsigned char buffer[width*height];
    int** image = (int**)malloc(sizeof(int*)*height);
    
    size_t ligne_size = sizeof(int) * width;

    fread(buffer, sizeof(buffer), 1, ptr);

    for (int i=0; i<height; i++) {
        int* ligne = (int*)malloc(ligne_size);
        for (int j=0; j<width; j++) {
            ligne[j] = (int)buffer[j+i*width];
        }
        image[i] = ligne;
    }
    return image;
}

// renvoie [nb_elem, width, height]
int* read_mnist_images_parameters(char* filename) {
    int* tab = malloc(sizeof(int)*3);
    FILE *ptr;
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t number_of_images;
    unsigned int height;
    unsigned int width;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&number_of_images, sizeof(uint32_t), 1, ptr);
    tab[0] = swap_endian(number_of_images);

    fread(&height, sizeof(unsigned int), 1, ptr);
    tab[1] = swap_endian(height);

    fread(&width, sizeof(unsigned int), 1, ptr);
    tab[2] = swap_endian(width);

    return tab;
}

uint32_t read_mnist_labels_nb_images(char* filename) {
    FILE *ptr;
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t number_of_images;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&number_of_images, sizeof(uint32_t), 1, ptr);
    number_of_images = swap_endian(number_of_images);

    return number_of_images;
}

// Lit un set de données images sous format de la base de données MNIST
int*** read_mnist_images(char* filename) {
    FILE *ptr;
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t number_of_images;
    unsigned int height;
    unsigned int width;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&number_of_images, sizeof(uint32_t), 1, ptr);
    number_of_images = swap_endian(number_of_images);

    fread(&height, sizeof(unsigned int), 1, ptr);
    height = swap_endian(height);

    fread(&width, sizeof(unsigned int), 1, ptr);
    width = swap_endian(width);

    int*** tab = (int***)malloc(sizeof(int**)*number_of_images);

    for (int i=0; i < number_of_images; i++) {
        tab[i] = read_image(width, height, ptr);
    }
    return tab;
}

// Renvoie des labels formattés sous le format de la base MNIST
unsigned int* read_mnist_labels(char* filename) {
    FILE* ptr;

    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t number_of_items;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2049) {
        printf("Incorrect magic number !\n");
        exit(1);
    }

    fread(&number_of_items, sizeof(uint32_t), 1, ptr);
    number_of_items = swap_endian(number_of_items);

    unsigned char buffer[number_of_items];
    fread(buffer, sizeof(unsigned char), number_of_items, ptr);

    unsigned int* labels = malloc(sizeof(unsigned int)*number_of_items);

    for (int i=0; i< number_of_items; i++) {
        labels[i] = (unsigned int)buffer[i];
    }
    return labels;
}
