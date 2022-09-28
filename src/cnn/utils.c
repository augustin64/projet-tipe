#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <string.h>

#include "../colors.h"
#include "include/struct.h"

#define checkEquals(var, name, indice) if (network1->var != network2->var) { printf_error("network1->" name " et network2->" name " ne sont pas égaux\n"); if (indice != -1) {printf(BOLDBLUE"[ INFO_ ]"RESET" indice: %d\n", indice);} return false; }

bool equals_networks(Network* network1, Network* network2) {
    checkEquals(size, "size", -1);
    checkEquals(initialisation, "initialisation", -1);
    checkEquals(dropout, "dropout", -1);
    
    for (int i=0; i < network1->size; i++) {
        checkEquals(width[i], "input_width", i);
        checkEquals(depth[i], "input_depth", i);
    }

    for (int i=0; i < network1->size; i++) {
        checkEquals(kernel[i]->activation, "kernel[i]->activation", i);
        if ((!network1->kernel[i]->cnn ^ !network2->kernel[i]->cnn) || (!network1->kernel[i]->nn ^ !network2->kernel[i]->nn)) {
            printf(BOLDRED "[ ERROR ]" RESET "network1->kernel[%d] et network1->kernel[%d] diffèrent de type\n", i, i);
            return false;
        }
        if (!network1->kernel[i]->cnn && !network1->kernel[i]->nn) {
            // Type Pooling
            // checkEquals(kernel[i]->linearisation, "kernel[i]->linearisation", i);
        } else if (!network1->kernel[i]->cnn) {
            // Type NN
            checkEquals(kernel[i]->nn->input_units, "kernel[i]->nn->input_units", i);
            checkEquals(kernel[i]->nn->output_units, "kernel[i]->nn->output_units", i);
            for (int j=0; j < network1->kernel[i]->nn->output_units; j++) {
                checkEquals(kernel[i]->nn->bias[j], "kernel[i]->nn->bias[j]", j);
            }
            for (int j=0; j < network1->kernel[i]->nn->input_units; j++) {
                for (int k=0; k < network1->kernel[i]->nn->output_units; k++) {
                    checkEquals(kernel[i]->nn->weights[j][k], "kernel[i]->nn->weights[j][k]", k);
                }
            }
        } else {
            // Type CNN
            checkEquals(kernel[i]->cnn->k_size, "kernel[i]->k_size", i);
            checkEquals(kernel[i]->cnn->rows, "kernel[i]->rows", i);
            checkEquals(kernel[i]->cnn->columns, "kernel[i]->columns", i);
            for (int j=0; j < network1->kernel[i]->cnn->columns; j++) {
                for (int k=0; k < network1->kernel[i]->cnn->k_size; k++) {
                    for (int l=0; l < network1->kernel[i]->cnn->k_size; l++) {
                        checkEquals(kernel[i]->cnn->bias[j][k][l], "kernel[i]->cnn->bias[j][k][l]", l);
                    }
                }
            }
            for (int j=0; j < network1->kernel[i]->cnn->rows; j++) {
                for (int k=0; k < network1->kernel[i]->cnn->columns; k++) {
                    for (int l=0; l < network1->kernel[i]->cnn->k_size; l++) {
                        for (int m=0; m < network1->kernel[i]->cnn->k_size; m++) {
                            checkEquals(kernel[i]->cnn->w[j][k][l][m], "kernel[i]->cnn->bias[j][k][l][m]", m);
                        }
                    }
                }
            }
        }
    }

    return true;
}