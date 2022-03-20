#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>


uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


// Prévisualise un chiffre écrit à la main
// de taille width x height
// commencant à l'adresse mémoire start
// dans le fichier pointé par ptr
void print_image(unsigned int width, unsigned int height, FILE* ptr, int start) {
    unsigned char buffer[width*height];

    fread(buffer, sizeof(buffer), 1, ptr);

    int cur;
    char tab[] = {' ', '.', ':', '%', '#', '\0'};

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            cur = (int)buffer[j+i*width];
            printf("%c", tab[cur/52]);
        }
        printf("\n");
    }
}

// Lit un set de données images de la base de données MNIST
// dans le fichier situé à filename, les
// images comportant comme labels 'labels'
void read_mnist_images(char* filename, unsigned int* labels) {
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

    //printf("magic number: %" PRIu32 "\n", magic_number);
    //printf("number of images: %" PRIu32 "\n", number_of_images);
    //printf("%u x %u\n\n", width, height);

    for (int i=0; i < number_of_images; i++) {
        printf("--- Number %d : %d ---\n", i, labels[i]);
        print_image(width, height, ptr, (i*width*height));
    }
}


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

    printf("number of items: %" PRIu32 "\n", number_of_items);

    unsigned char buffer[number_of_items];
    fread(buffer, sizeof(unsigned char), number_of_items, ptr);

    unsigned int* labels = malloc(sizeof(unsigned int)*number_of_items);

    for (int i=0; i< number_of_items; i++) {
        labels[i] = (unsigned int)buffer[i];
    }
    return labels;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Utilisation: %s [IMAGES FILE] [LABELS FILE]\n", argv[0]);
        return 1;
    }
    unsigned int* labels = read_mnist_labels(argv[2]);
    read_mnist_images(argv[1], labels);
    free(labels);
    return 0;
}
