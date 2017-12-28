#include <stdio.h>
#include <book.h>

int main(void)
{
	cudaDeviceProp prop;

	int count;
	HANDLE_ERROR(cudaGetDeviceCount(&count));

	for (int i = 0; i < count; i++)
	{
		HANDLE_ERROR(cudaGetDeviceProperties(&prop, i));
		printf_s("  --- General Information for device %d ---\n", i);
		printf_s("Name:  %s\n", prop.name);
		printf_s("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf_s("Clock rate (KHz):  %d\n", prop.clockRate);
		printf_s("Device copy overlap:  ");
		if (prop.deviceOverlap)
			printf_s("Enabled\n");
		else
			printf_s("Disabled\n");
		printf_s("\n");
		printf_s("  --- Memory Information for device %d ---\n", i);
		printf_s("Total global memory:  %llu\n", prop.totalGlobalMem);
		printf_s("Total constant memory:  %ld\n", prop.totalConstMem);
		printf_s("Memory bus width:  %d\n", prop.memoryBusWidth);
		printf_s("ECC memory enabled:  ");
		if (prop.ECCEnabled)
			printf_s("Enabled\n");
		else
			printf_s("Disabbled\n");
		printf_s("\n");
		printf_s("  ---Mulitprocessor Information for device %d ---\n", i);
		printf_s("Multiprocessor count:  %d\n", prop.multiProcessorCount);
		printf_s("Shared memory per mp:  %d\n", prop.sharedMemPerMultiprocessor);
		printf_s("Shared memory per block:  %ld\n", prop.sharedMemPerBlock);
		printf_s("Max threads per block:  %d\n", prop.maxThreadsPerBlock);

		printf_s("\n");

	}

	return 0;
}