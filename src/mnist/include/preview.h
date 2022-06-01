#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <inttypes.h>

#ifndef DEF_PREVIEW_H
#define DEF_PREVIEW_H

void print_image(unsigned int width, unsigned int height, int** image);
void preview_images(char* images_file, char* labels_file);

#endif

               