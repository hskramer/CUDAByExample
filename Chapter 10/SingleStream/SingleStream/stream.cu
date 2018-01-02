
#include <book.h>

#define N   (1024 * 1024)
#define FULL_DATA_SIZE   (N * 20)


__global__ void kernel(int *a, int *b, int *c)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < N)
	{
		int idx1 = (idx + 1) % 256;
		int idx2 = (idx + 2) % 256;
		float   as = (a[idx] + a[idx1] + a[idx2]) / 3.0f;
		float   bs = (b[idx] + b[idx1] + b[idx2]) / 3.0f;
		c[idx] = (as + bs) / 2;
	}
}


int main(void)
{
	cudaEvent_t     start, stop;
	float           elapsedTime;

	cudaStream_t    stream;
	int *h_a, *h_b, *h_c;
	int *d_a, *d_b, *d_c;

	// start the timers
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// initialize the stream
	HANDLE_ERROR(cudaStreamCreate(&stream));

	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&d_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c, N * sizeof(int)));

	// allocate host locked memory, used to stream
	HANDLE_ERROR(cudaHostAlloc((void**)&h_a, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_b, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));
	HANDLE_ERROR(cudaHostAlloc((void**)&h_c, FULL_DATA_SIZE * sizeof(int), cudaHostAllocDefault));

	for (int i = 0; i<FULL_DATA_SIZE; i++)
	{
		h_a[i] = rand();
		h_b[i] = rand();
	}

	HANDLE_ERROR(cudaEventRecord(start, 0));
	// now loop over full data, in bite-sized chunks
	for (int i = 0; i<FULL_DATA_SIZE; i += N)
	{
		// copy the locked memory to the device, async
		HANDLE_ERROR(cudaMemcpyAsync(d_a, h_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));
		HANDLE_ERROR(cudaMemcpyAsync(d_b, h_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream));

		kernel << <N / 256, 256, 0, stream >> >(d_a, d_b, d_c);

		// copy the data from device to locked memory
		HANDLE_ERROR(cudaMemcpyAsync(h_c + i, d_c, N * sizeof(int), cudaMemcpyDeviceToHost, stream));

	}
	// copy result chunk from locked to full buffer
	HANDLE_ERROR(cudaStreamSynchronize(stream));

	HANDLE_ERROR(cudaEventRecord(stop, 0));

	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	printf("Time taken:  %3.2f ms\n", elapsedTime);

	// cleanup the streams and memory
	HANDLE_ERROR(cudaFreeHost(h_a));
	HANDLE_ERROR(cudaFreeHost(h_b));
	HANDLE_ERROR(cudaFreeHost(h_c));
	HANDLE_ERROR(cudaFree(d_a));
	HANDLE_ERROR(cudaFree(d_b));
	HANDLE_ERROR(cudaFree(d_c));
	HANDLE_ERROR(cudaStreamDestroy(stream));

	return 0;
}

