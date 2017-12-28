#include <book.h>

#define imin(a,b)	(a < b ? a:b)

const int	N = 33 * 1024;
const int	threadsPerBlock = 256;
const int	blocksPerGrid = imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

__global__ void dot(float *a, float *b, float *c)
{
	__shared__ float cache[threadsPerBlock];

	int		tid = blockIdx.x * blockDim.x + threadIdx.x;
	int		cacheIndex = threadIdx.x;

	float	temp = 0;
	while (tid < N)
	{
		temp += a[tid] * b[tid];
		tid += blockDim.x * gridDim.x;
	}
	// set cache values
	cache[cacheIndex] = temp;

	// synchronize threads in this block
	__syncthreads();

	// for reductions, threadsPerBlock must be a power of  2 because of the following code
	int		i = blockDim.x / 2;
	while (i != 0)
	{
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;		
	}

	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];
}

int main(void)
{
	float	*a, *b, c, *partial_c;
	float	*d_a, *d_b, *d_partial_c;

	// allocate memory on the CPU side
	a = (float *)malloc(N * sizeof(float));
	b = (float *)malloc(N * sizeof(float));
	partial_c = (float *)malloc(blocksPerGrid * sizeof(float));

	// allocate memory on the GPU
	a = (float *)malloc(N * sizeof(float));
	HANDLE_ERROR(cudaMalloc((void **)&d_a, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&d_b, N * sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void **)&d_partial_c, blocksPerGrid * sizeof(float)));

	// fill in the host memory with data
	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaMemcpy(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, b, N * sizeof(float), cudaMemcpyHostToDevice));

	dot << <blocksPerGrid, threadsPerBlock >> > (d_a, d_b, d_partial_c);

	// copt the array 'c' back from the GPU to the CPU
	HANDLE_ERROR(cudaMemcpy(partial_c, d_partial_c, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

	// finish up on the CPU side
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	#define sum_squares(x) (x*(x+1)*(2*x+1)/6)
	printf("Does GPU value %.6g = %.6g?\n", c, 2 * sum_squares((float)(N - 1)));

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_partial_c);

	free(a);
	free(b);
	free(partial_c);

	
}