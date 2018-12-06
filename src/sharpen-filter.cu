#include <stdio.h>
#include <stdlib.h>
#include <nvToolsExt.h>
extern "C" {
	#include "pgm.h"
}

__host__ __device__
bool pixel_out_of_bounds(struct pgm *img, int i, int j)
{
	return i < 0 || j < 0 || i >= img->height || j >= img->width;
}

__host__ __device__
int get_pixel_at(struct pgm *img, int i, int j)
{
	if (pixel_out_of_bounds(img, i, j))
		return 0;
	return img->pixels[img->width * i + j];
}

__host__ __device__
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

	return sum < 0 ? 0 : sum > img->maxval ? img->maxval : sum;
}

__host__
struct pgm copy_pgm_to_gpu(struct pgm *img)
{
	size_t size = img->width * img->height * sizeof(int);
	struct pgm gpu_img;

	gpu_img.width = img->width;
	gpu_img.height = img->height;
	gpu_img.maxval = img->maxval;
	cudaMalloc(&gpu_img.pixels, size);
	if (gpu_img.pixels != NULL)
		cudaMemcpy(gpu_img.pixels, img->pixels, size, cudaMemcpyHostToDevice);

	return gpu_img;
}

__global__
void sharpen_filter_kernel(struct pgm img, struct pgm original_img)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y;
	int j = blockIdx.x * blockDim.x + threadIdx.x;

	if (i < img.height && j < img.width)
		img.pixels[i * img.width + j] = pixel_sharpen_filter(&original_img, i, j);
}

__host__
void sharpen_filter_on_gpu(struct pgm *img)
{
	struct pgm gpu_img = copy_pgm_to_gpu(img);
	struct pgm gpu_original_img = copy_pgm_to_gpu(img);

	dim3 dimBlock(32, 32);
	dim3 dimGrid(img->width / dimBlock.x, img->height / dimBlock.y);
	sharpen_filter_kernel<<<dimGrid, dimBlock>>>(gpu_img, gpu_original_img);

	// copy result back to host
	size_t size = img->width * img->height * sizeof(int);
	cudaMemcpy(img->pixels, gpu_img.pixels, size, cudaMemcpyDeviceToHost);

	cudaFree(gpu_img.pixels);
	cudaFree(gpu_original_img.pixels);
}

// Apply pixel_sharpen_filter for every pixel of img
__host__
void sharpen_filter_on_cpu(struct pgm *img)
{
	struct pgm *original_img = copy_pgm(img);

	nvtxRangePush(__FUNCTION__);
	for (int i = 0; i < img->height;  i++) {
		for (int j = 0; j < img->width; j++) {
			img->pixels[i * img->width + j] = pixel_sharpen_filter(original_img, i, j);
		}
	}
	nvtxRangePop();

	free(original_img);
}

int main(int argc, char **argv)
{
	nvtxRangePush(__FUNCTION__);
	struct pgm *img = read_pgm("baboon.bin.pgm");
	struct pgm *cuda_img = copy_pgm(img);

	sharpen_filter_on_gpu(cuda_img);
	save_pgm(cuda_img, "baboon_sharpen.cuda.bin.pgm");

	sharpen_filter_on_cpu(img);
	save_pgm(img, "baboon_sharpen.bin.pgm");

	free(img);
	free(cuda_img);
	nvtxRangePop();
	return 0;
}
