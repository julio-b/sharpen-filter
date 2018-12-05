#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <nvToolsExt.h>
#include "pgm.h"
#include "utils.h"

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
	nvtxRangePush(__FUNCTION__);
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

	image = (struct pgm *) nmalloc(pgm_buffer_size(width, height));
	if (image == NULL)
		return NULL;

	image->width = width;
	image->height = height;
	image->maxval = maxval;
	image->pixels = (int *) image + sizeof(struct pgm);

	// read image pixels
	int *ptr = image->pixels;
	while ((c = fgetc(file_in)) != EOF) {
	    *ptr++ = c;
	}

	fclose(file_in);
	nvtxRangePop();
	return image;
}

// Returns a deep copy of img
struct pgm *copy_pgm(struct pgm *img)
{
	nvtxRangePush(__FUNCTION__);
	size_t img_size = pgm_buffer_size(img->width, img->height);
	struct pgm *copyimg = (struct pgm *) nmalloc(img_size);

	if (copyimg == NULL)
		return NULL;
	nmemcpy(copyimg, img, img_size);
	copyimg->pixels = (int *) copyimg + sizeof(struct pgm);

	nvtxRangePop();
	return copyimg;
}


bool save_pgm(struct pgm *img, char *filename)
{
	nvtxRangePush(__FUNCTION__);
	FILE *file_out;
	int *ptr;
	int *end;

	file_out = fopen(filename, "wb");
	if (file_out == NULL)
		return false;

	fprintf(file_out, "P5\n%d %d\n%d\n", img->width, img->height, img->maxval);

	ptr = img->pixels;
	end = img->pixels + img->width * img->height;
	while (ptr != end)
		if (fputc(*ptr++, file_out) == EOF)
			return false;

	fclose(file_out);
	nvtxRangePop();
	return true;
}
