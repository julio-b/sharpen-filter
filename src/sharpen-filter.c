#include <stdio.h>
#include <stdlib.h>
#include "pgm.h"

int main(int argc, char **argv)
{
	struct pgm *img = read_pgm("../sample-imgs/baboon.bin.pgm");
	if(img == NULL)
		return -1;

	for (int i = 0; i < img->height; i++) {
		for (int j = 0; j < img->width; j++) {
			printf("%02x ", *(img->pixels + img->width * i + j));
		}
		puts("\n");
	}

	free(img);
	return 0;
}
