#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>


uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


void print_image(unsigned int width, unsigned int height, FILE* ptr, int start) {
    unsigned char buffer[width*height+start];

    fread(buffer, sizeof(buffer), 1, ptr);

    int cur;
    char tab[] = {' ', '.', ':', '%', '#', '\0'};

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            cur = (int)buffer[start+j+i*width];
            printf("%c", tab[cur/52]);
        }
        printf("\n");
    }
}


void read_mnist_images(char* filename) {
    unsigned char buffer[4];
    FILE *ptr;
    
    ptr = fopen(filename, "rb");

    uint32_t magic_number;
    uint32_t number_of_images;
    unsigned int height;
    unsigned int width;

    fread(&magic_number, sizeof(uint32_t), 1, ptr);
    magic_number = swap_endian(magic_number);

    if (magic_number != 2051){
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
        printf("--- Number %d ---\n", i);
        print_image(width, height, ptr, (i*width*height));
    }
}


int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("Utilisation: %s [FILE]\n", argv[0]);
        return 1;
    }
    read_mnist_images(argv[1]);
    return 0;
}
