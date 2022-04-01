#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef DEF_PREVIEW_H
#define DEF_PREVIEW_H

uint32_t swap_endian(uint32_t val);
void print_image(unsigned int width, unsigned int height, FILE* ptr, int start);
void read_mnist_images(char* filename, unsigned int* labels);
unsigned int* read_mnist_labels(char* filename);

#endif

               