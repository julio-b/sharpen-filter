#include <stdio.h>
#include <stdlib.h>
#include "pgm.h"

bool pixel_out_of_bounds(struct pgm *img, int i, int j)
{
	return i < 0 || j < 0 || i >= img->height || j >= img->width;
}

int get_pixel_at(struct pgm *img, int i, int j)
{
	if (pixel_out_of_bounds(img, i, j))
		return 0;
	return img->pixels[img->width * i + j];
}

int pixel_sharpen_filter(struct pgm *img, int i, int j)
{
	int sum = 0;
	static int filter[] = { -1,  -1,  -1,
	                        -1,  +9,  -1,
	                        -1,  -1,  -1 };
	int pixels_block[9];

	pixels_block[0] = get_pixel_at(img, i-1, j-1);
	pixels_block[1] = get_pixel_at(img, i-1, j  );
	pixels_block[2] = get_pixel_at(img, i-1, j+1);
	pixels_block[3] = get_pixel_at(img, i  , j-1);
	pixels_block[4] = get_pixel_at(img, i  , j  );
	pixels_block[5] = get_pixel_at(img, i  , j+1);
	pixels_block[6] = get_pixel_at(img, i+1, j-1);
	pixels_block[7] = get_pixel_at(img, i+1, j  );
	pixels_block[8] = get_pixel_at(img, i+1, j+1);

	for (int q = 0; q < 9; q++)
		sum += filter[q] * pixels_block[q];

	return sum;
}

// Apply pixel_sharpen_filter for every pixel of img
bool sharpen_filter(struct pgm *img)
{
	struct pgm *original_img = copy_pgm(img);
	if (original_img == NULL)
		return false;

	for (int i = 0; i < img->height;  i++) {
		for (int j = 0; j < img->width; j++) {
			img->pixels[i * img->width + j] =
				pixel_sharpen_filter(original_img, i, j);
		}
	}

	free(original_img);
	return true;
}

int main(int argc, char **argv)
{
	struct pgm *img = read_pgm("../sample-imgs/baboon.bin.pgm");
	if (img == NULL)
		return -1;

	sharpen_filter(img);

	if (!save_pgm(img, "/tmp/sharpen_baboon.bin.pgm"))
		return -2;

	free(img);
	return 0;
}
