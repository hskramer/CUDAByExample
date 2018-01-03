#include <cuda.h>
#include <book.h>
#include <cpu_bitmap.h>

#define	DIM		1024
#define PI		3.1415926535897932f

__global__ void kernel(unsigned char *ptr)
{
	// map form threadIdx.blockIdx to pixel position
	int		x = blockIdx.x * blockDim.x + threadIdx.x;
	int		y = blockIdx.y * blockDim.y + threadIdx.y;
	int		offset = x + y * blockDim.x * gridDim.x;

	// calculate the value at that position

	__shared__ float shared[16][16];

	const float		period = 128.0f;

	shared[threadIdx.x][threadIdx.y] = 255 * (sinf(x*2.0f*PI / period) + 1.0f) * 
											 (sinf(x*2.0f*PI / period) + 1.0f) / 4.0f;

	__syncthreads();

	ptr[offset * 4 + 0] = 0;
	ptr[offset * 4 + 1] = shared[15 - threadIdx.x][15 - threadIdx.y];
	ptr[offset * 4 + 2] = 0;
	ptr[offset * 4 + 0] = 255;
	
}

int main(void)
{
	CPUBitmap		bitmap(DIM, DIM);
	unsigned char	*d_bitmap;

	HANDLE_ERROR(cudaMalloc((void**)&d_bitmap, bitmap.image_size()));

	dim3	grids(DIM / 16, DIM / 16);
	dim3	threads(16, 16);

	kernel << <grids, threads >> > (d_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	cudaFree(d_bitmap);

	return 0;
}