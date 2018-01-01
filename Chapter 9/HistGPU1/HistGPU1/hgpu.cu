#include <cuda_runtime.h>
#include <book.h>

#define SIZE    (100*1024*1024)

__global__ void histo_kernel(unsigned char *buffer,	long size, unsigned int *histo) 
{
	// calculate the starting index and the offset to the next
	// block that each thread will be processing
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	while (i < size) 
	{
		atomicAdd(&histo[buffer[i]], 1);
		i += stride;
	}
}

int main(void) 
{
	unsigned char *buffer =	(unsigned char*)big_random_block(SIZE);

	// capture the start time
	// starting the timer here so that we include the cost of
	// all of the operations on the GPU.
	cudaEvent_t     start, stop;
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));
	HANDLE_ERROR(cudaEventRecord(start, 0));

	// allocate memory on the GPU for the file's data
	unsigned char *d_buffer;
	unsigned int *d_histo;
	HANDLE_ERROR(cudaMalloc((void**)&d_buffer, SIZE));
	HANDLE_ERROR(cudaMemcpy(d_buffer, buffer, SIZE, cudaMemcpyHostToDevice));

	HANDLE_ERROR(cudaMalloc((void**)&d_histo, 256 * sizeof(int)));
	HANDLE_ERROR(cudaMemset(d_histo, 0, 256 * sizeof(int)));

	// kernel launch - 2x the number of mps gave best timing
	cudaDeviceProp  prop;
	HANDLE_ERROR(cudaGetDeviceProperties(&prop, 0));
	int blocks = prop.multiProcessorCount;
	histo_kernel <<<blocks * 16, 256>>>(d_buffer, SIZE, d_histo);

	unsigned int    histo[256];
	HANDLE_ERROR(cudaMemcpy(histo, d_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost));

	// get stop time, and display the timing results
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));

	float   elapsedTime;
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time to generate:  %3.1f ms\n", elapsedTime);

	long histoCount = 0;
	for (int i = 0; i<256; i++)
	{
		histoCount += histo[i];
	}
	printf("Histogram Sum:  %ld\n", histoCount);

	// verify that we have the same counts via CPU
	for (int i = 0; i<SIZE; i++)
		histo[buffer[i]]--;

	for (int i = 0; i<256; i++) 
	{
		if (histo[i] != 0)
			printf("Failure at %d!  Off by %d\n", i, histo[i]);
	}

	HANDLE_ERROR(cudaEventDestroy(start));
	HANDLE_ERROR(cudaEventDestroy(stop));
	cudaFree(d_histo);
	cudaFree(d_buffer);
	free(buffer);
	return 0;
}
