#ifndef DEF_TEST_H
#define DEF_TEST_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#include "../../src/common/include/colors.h"

#define _TEST_PRESENTATION(description) { printf("\n" BLUE "#### %s:" BOLD "%s" RESET BLUE " #####" RESET "\n", __FILE__, description); }

#define _TEST_ASSERT(condition, description) {                                              \
    if (condition) {                                                                        \
        printf("[" GREEN "OK" RESET "] %s:%d: %s\n", __FILE__, __LINE__, description);      \
    } else {                                                                                \
        printf("[" RED "ERREUR" RESET "] %s:%d: %s\n", __FILE__, __LINE__, description);    \
        exit(1);                                                                            \
    }                                                                                       \
}



#endif