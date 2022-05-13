#define PADING_INPUT 2

void write_image_in_newtork_32(int** image, int height, int width, float** network) {
    /* Ecrit une image 28*28 au centre d'un tableau 32*32 et met à 0 le reste */

    for (int i=0; i < height+2*PADING_INPUT; i++) {
        for (int j=PADING_INPUT; j < width+2*PADING_INPUT; j++) {
            if (i<PADING_INPUT || i>height+PADING_INPUT || j<PADING_INPUT || j>width+PADING_INPUT){
                network[i][j] = 0.;
            }
            else {
                network[i][j] = (float)image[i][j] / 255.0f;
            }
        }
    }
}
