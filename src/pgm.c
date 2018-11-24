#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "pgm.h"

#define BUFSIZE 30

int fgetint_ascii(FILE *file)
{
	static char buf[BUFSIZE];
	int c;
	int i = 0;

	while (i < BUFSIZE - 1  && (c = fgetc(file)) != EOF && !isspace(c)) {
		buf[i++] = c;
	}
	buf[i] = '\0';

	return strtol(buf, NULL, 10);
}

size_t pgm_buffer_size(int width, int height)
{
	return sizeof(struct pgm) + sizeof(int) * width * height;
}

struct pgm *read_pgm(char *filename)
{
	FILE *file_in;
	int width;
	int height;
	int maxval;
	struct pgm *image = NULL;
	int c;

	file_in = fopen(filename, "rb");
	if (file_in == NULL)
		return NULL;

	if (fgetc(file_in) != 'P' || fgetc(file_in) != '5')
		return NULL;

	fgetc(file_in); // whitespace

	// ignore comment line
	if ((c = fgetc(file_in)) == '#') {
		while (fgetc(file_in) != '\n');
	} else {
		ungetc(c, file_in);
	}

	width = fgetint_ascii(file_in);
	height = fgetint_ascii(file_in);
	maxval = fgetint_ascii(file_in);

	image = (struct pgm *) malloc(pgm_buffer_size(width, height));
	if (image == NULL)
		return NULL;

	image->width = width;
	image->height = height;
	image->maxval = maxval;
	image->pixels = (int *) image + sizeof(struct pgm);

	// read image pixels
	int *ptr = image->pixels;
	while((c = fgetc(file_in)) != EOF) {
	    *ptr++ = c;
	}

	fclose(file_in);
	return image;
}
