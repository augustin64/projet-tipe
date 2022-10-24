#include <stdio.h>

#include "include/colors.h"

void printf_error(char* string) {
    printf(BOLDRED "[ ERROR ]" RESET " %s", string);
}

void printf_warning(char* string) {
    printf(BOLDYELLOW "[WARNING]" RESET " %s", string);
}

void printf_info(char* string) {
    printf(BOLDBLUE "[ INFO  ]" RESET " %s", string);
}