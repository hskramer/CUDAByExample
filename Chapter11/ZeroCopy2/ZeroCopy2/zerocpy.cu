#include <book.h>



int main()
{

	float           *a, *b, c, *partial_c;
	float           *d_a, *d_b, *d_partial_c;
	float           elapsedTime;

	HANDLE_ERROR(cudaEventCreate(&start));
	HANDLE_ERROR(cudaEventCreate(&stop));

	// allocate the memory on the CPU
	HANDLE_ERROR(cudaHostAlloc((void**)&a, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&b, size * sizeof(float), cudaHostAllocWriteCombined | cudaHostAllocMapped));
	HANDLE_ERROR(cudaHostAlloc((void**)&partial_c, blocksPerGrid * sizeof(float), cudaHostAllocMapped));

	// find out the GPU pointers
	HANDLE_ERROR(cudaHostGetDevicePointer(&d_a, a, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&d_b, b, 0));
	HANDLE_ERROR(cudaHostGetDevicePointer(&d_partial_c, partial_c, 0));

	// fill in the host memory with data
	for (int i = 0; i < size; i++)
	{
		a[i] = i;
		b[i] = i * 2;
	}

	HANDLE_ERROR(cudaEventRecord(start, 0));

	dot << <blocksPerGrid, threadsPerBlock >> >(size, d_a, d_b, d_partial_c);

	HANDLE_ERROR(cudaThreadSynchronize());
	HANDLE_ERROR(cudaEventRecord(stop, 0));
	HANDLE_ERROR(cudaEventSynchronize(stop));
	HANDLE_ERROR(cudaEventElapsedTime(&elapsedTime, start, stop));

	// finish up on the CPU side
	c = 0;
	for (int i = 0; i < blocksPerGrid; i++)
	{
		c += partial_c[i];
	}

	HANDLE_ERROR(cudaFreeHost(a));
	HANDLE_ERROR(cudaFreeHost(b));
	HANDLE_ERROR(cudaFreeHost(partial_c));

	const int num_streams = 8;

	cudaStream_t streams[num_streams];
	float *data[num_streams];

	for (int i = 0; i < num_streams; i++)
	{
		cudaStreamCreate(&streams[i]);

		cudaMalloc(&data[i], N * sizeof(float));

		// launch one worker kernel per stream
		kernel << <1, 64, 0, streams[i] >> >(data[i], N);

		// launch a dummy kernel on the default stream
		kernel << <1, 1 >> >(0, 0);
	}

	cudaDeviceReset();

	return 0;

}