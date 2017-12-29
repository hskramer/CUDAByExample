



#define DIM	1024
#define PI 3.1415926535897932f
#define MAX_TEMP 1.0f
#define MIN_TEMP 0.0001f
#define SPEED	0.25f

__global__ void copy_const_ kernel(float *iptr, const float *cptr)
{
	// map from threadIdx/blockIdx to pixel position
	int	x = blockIdx.x * blockDim.x + threadIdx.x;
	int	y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y *blockDim.x*gridDim.x;

	if (cptr[offset] != 0)	iptr[offset] = cptr[offset];
}

__global__ void blend_kernel(float *outSrc, const float *inSrc)
{
	int	x = blockIdx.x * blockDim.x + threadIdx.x;
	int	y = blockIdx.y * blockDim.y + threadIdx.y;
	int offset = x + y *blockDim.x*gridDim.x;

	int	left = offset - 1;
	int right = offset + 1;
	if (x == 0)		left++;
	if (x == DIM - 1)	  right--;

	int top = offset - DIM;
	int bottom = offset + DIM;
	if (y == 0)		top += DIM;
	if (y == DIM - 1) bottom -= DIM;

	outSrc[offset] = inSrc[offset] + SPEED * (inSrc[top]) + inSrc[bottom] + inSrc[left] + inSrc[right] - inSrc[offset] * 4);
}

struct DataBlock {
	unsigned char	*ouput_bitmap;
	float			*d_inSrc;
	float			*d_outSrc;
	float			*d_constSrc;
	CPUAnimBitMap	*bitmap;
	cudaEvent_t		start, stop;
	float			totalTime;
	float			frames;
};

void anim_gpu(DataBlock *d, int ticks)
{
	HANDLE_ERROR(cudaEventRecord(d->start, 0));
	dim3	blocks(DIM / 16, DIM / 16);
	dim3	threads(16, 16);
	CPUAnimBitmap	*bitmap = d->bitmap;

	for (int = 0; i < 90; i++)
	{
		copy_const_kernel << <blocks, threads >> > (d->d_inSrc, d->d_constSrc);

		blend_kernel << <blocks, threads >> > (d->d_outSrc, d->inSrc);

		swap(d->d_inSrc, d->d_outSrc);
	}

	float_to_color << <blocks, threads >> > (d->output_bitmap, d->d_inSrc);

	HANDLE_ERROR
}