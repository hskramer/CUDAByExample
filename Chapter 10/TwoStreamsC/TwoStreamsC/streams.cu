#include <book.h>

#define N   (1024 * 1024)
#define FULL_DATA_SIZE   (N * 20)


__global__ void kernel(int *a, int *b, int *c)
{
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
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

	cudaStream_t    stream0, stream1;
	int *h_a, *h_b, *h_c;
	int *d_a0, *d_b0, *d_c0;
	int *d_a1, *d_b1, *d_c1;

	// start the timers
	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// initialize the streams
	HANDLE_ERROR(cudaStreamCreate(&stream0));
	HANDLE_ERROR(cudaStreamCreate(&stream1));

	// allocate the memory on the GPU
	HANDLE_ERROR(cudaMalloc((void**)&d_a0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c0, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_a1, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b1, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c1, N * sizeof(int)));

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
	for (int i = 0; i<FULL_DATA_SIZE; i += N * 2)
	{
		// enqueue copies of a in stream0 and stream1
	
		HANDLE_ERROR(cudaMemcpyAsync(d_a0, h_a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(d_a1, h_a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

		// enqueue copies of b in stream0 and stream1
	
		HANDLE_ERROR(cudaMemcpyAsync(d_b0, h_b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(d_b1, h_b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1));

		// enqueue kernels in stream0 and stream1   
	
		kernel<<<N / 256, 256, 0, stream0>>>(d_a0, d_b0, d_c0);
		kernel<<<N / 256, 256, 0, stream1 >>>(d_a1, d_b1, d_c1);

		// enqueue copies of c from device to locked memory

		HANDLE_ERROR(cudaMemcpyAsync(h_c + i, d_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0));
		HANDLE_ERROR(cudaMemcpyAsync(h_c + i + N, d_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1));
	}

	HANDLE_ERROR(cudaStreamSynchronize(stream0));
	HANDLE_ERROR(cudaStreamSynchronize(stream1));

	HANDLE_ERROR(cudaEventRecord(stop, 0));

	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("Time taken:  %3.2f ms\n", elapsedTime);

	// cleanup the streams and memory
	HANDLE_ERROR(cudaFreeHost(h_a));
	HANDLE_ERROR(cudaFreeHost(h_b));
	HANDLE_ERROR(cudaFreeHost(h_c));
	HANDLE_ERROR(cudaFree(d_a0));
	HANDLE_ERROR(cudaFree(d_b0));
	HANDLE_ERROR(cudaFree(d_c0));
	HANDLE_ERROR(cudaFree(d_a1));
	HANDLE_ERROR(cudaFree(d_b1));
	HANDLE_ERROR(cudaFree(d_c1));
	HANDLE_ERROR(cudaStreamDestroy(stream0));
	HANDLE_ERROR(cudaStreamDestroy(stream1));

	return 0;
}

