#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>


void print_image(unsigned int width, unsigned int height, FILE* ptr, int start) {
    unsigned char buffer[width*height+start];

    fread(buffer, sizeof(buffer), 1, ptr);

    int cur;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            cur = buffer[start+j+i*width];
            if (cur > 150)
                printf("0");
            else {
                if (cur > 100)
                    printf(".");
                else
                    printf(" ");
            }
        }
        printf("\n");
    }
    printf("\t\t---\n");
}



void print_bytes(char* filename) {
    unsigned char buffer[4];
    FILE *ptr;
    
    ptr = fopen(filename, "rb");

    fread(buffer, sizeof(buffer), 1, ptr);

    uint32_t magic_number = buffer[0];
    uint32_t number_of_images = (int)buffer[1];
    unsigned int height = buffer[2];
    unsigned int width = buffer[3];


    printf("magic number: %" PRIu32 "\n", magic_number);
    printf("number of images: %" PRIu32 "\n", number_of_images);
    printf("%x x %x\n\n", width, height);
    for (int i=0; i < number_of_images; i++)
        print_image(width, height, ptr, (i*width*height)+4);
}

int main(int argc, char *argv[]) {
    if (argc == 1) {
        printf("Utilisation: %s [FILE]\n", argv[0]);
        return 1;
    }
    print_bytes(argv[1]);
    return 0;
}
