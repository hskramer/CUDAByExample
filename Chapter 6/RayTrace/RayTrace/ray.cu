#include <cuda.h>
#include <book.h>
#include <cpu_bitmap.h>

#define	DIM		1024
#define INF		2e10f
#define SPHERES	20
#define rnd(x) (x * rand() / RAND_MAX)



struct Sphere {
	float	r, g, b;
	float	radius;
	float	x, y, z;

	__device__ float hit(float ox, float oy, float *n)
	{
		float	dx = ox - x;
		float	dy = oy - y;

		if (dx*dx + dy*dy < radius * radius)
		{
			float dz = sqrtf(radius * radius - dx * dx - dy * dy);
			*n = dz / sqrtf(radius * radius);
			return dz + z;
		}
		return -INF;
	}
};


__global__ void kernel(Sphere *s, unsigned char *ptr)
{
	int		x = blockIdx.x * blockDim.x + threadIdx.x;
	int		y = blockIdx.y * blockDim.y + threadIdx.y;
	int		offset = x + y * blockDim.x * gridDim.x;
	float	ox = (x - DIM / 2);
	float	oy = (y - DIM / 2);
	float	r = 0, g = 0, b = 0;
	float	maxz = -INF;

	for (int i = 0; i < SPHERES; i++)
	{
		float	n;
		float	t = s[i].hit(ox, oy, &n);

		if (t > maxz)
		{
			float	fscale = n;
			r = s[i].r * fscale;
			g = s[i].g * fscale;
			b = s[i].b * fscale;

		}
	}

	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;



}


int main(void)
{
	cudaEvent_t		start, stop;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	CPUBitmap	bitmap(DIM, DIM);
	unsigned  char		*d_bitmap;
	Sphere		*s;

	// allocate memory on GPU for output bitmap
	HANDLE_ERROR(cudaMalloc((void**)&d_bitmap, bitmap.image_size()));
	// allocate memory for Sphere dataset
	HANDLE_ERROR(cudaMalloc((void**)&s, sizeof(Sphere) * SPHERES));

	// allocate temp memory, initialize it, copy to GPU memory, then free temp memory
	Sphere	*temp_s = (Sphere *)malloc(sizeof(Sphere) * SPHERES);
	for (int i = 0; i < SPHERES; i++)
	{
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f) - 500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f) + 20;

		
	}

	HANDLE_ERROR(cudaMemcpy(s, temp_s, sizeof(Sphere) * SPHERES, cudaMemcpyHostToDevice));
	free(temp_s);

	dim3	grids(DIM / 16, DIM / 16);
	dim3	threads(16, 16);

	kernel << <grids, threads >> > (s,d_bitmap);

	HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), d_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost));

	bitmap.display_and_exit();
	cudaFree(s);

	return 0;
}