#include <stdio.h>
#include <book.h>

#define N	50000

__global__
void add(int *a, int *b, int *c)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < N)
	{
		c[i] = a[i] + b[i];
		i += blockDim.x*gridDim.x;
	}
}
int main(void)
{
	int a[N], b[N], c[N];
	int *d_a, *d_b, *d_c;

	HANDLE_ERROR(cudaMalloc((void**)&d_a, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_b, N * sizeof(int)));
	HANDLE_ERROR(cudaMalloc((void**)&d_c, N * sizeof(int)));

	for (int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i * i;
	}

	HANDLE_ERROR(cudaMemcpy(d_a, a, N * sizeof(int), cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_b, b, N * sizeof(int), cudaMemcpyHostToDevice));

	add <<<256, 256>> > (d_a, d_b, d_c);

	HANDLE_ERROR(cudaMemcpy(c, d_c, N * sizeof(int), cudaMemcpyDeviceToHost));

	bool success = true;
	for (int i = 0; i < N; i++)
	{
		if ((a[i] + b[i]) != c[i])
		{
			printf_s("Error: %d + %d != %d", a[i], b[i], c[i]);
			success = false;
		}
	}

	if (success)
	{
		for (int i = 0; i < 1000; i++)
		{
			printf_s("%d + %d = %d\n", b[i], a[i], c[i]);
		}
	}

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;

}