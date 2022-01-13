#include <iostream>
#include <math.h>
#include <algorithm>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#pragma comment(lib,"cuda.lib")
#pragma comment(lib,"cublas.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"cudadevrt.lib")
#pragma comment(lib,"cudart.lib")
#pragma comment(lib,"cudart_static.lib")
#pragma comment(lib,"curand.lib")

#define N (1000000)
#define BLOCK_SIZE (128)

using namespace std;


__global__ void find_max_min_kernel(int *A,int *result_min,int *result_max, int n, int bsize)
{
	__shared__ int b_min[BLOCK_SIZE];
	__shared__ int b_max[BLOCK_SIZE];
	//copy data to share memory
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	b_min[threadIdx.x] = INT_MAX;
	b_max[threadIdx.x] = INT_MIN;
	if (index < n) {
		b_min[threadIdx.x] = A[index];
		b_max[threadIdx.x] = A[index];
	}
	__syncthreads();

	// find min max loop
	for (int bs = BLOCK_SIZE / 2; bs > 0; bs /= 2)
	{
		int tmp_min;
		int tmp_max;
		if (threadIdx.x < bs) {
			tmp_min = min(b_min[threadIdx.x], b_min[threadIdx.x + bs]);
			tmp_max = max(b_max[threadIdx.x], b_max[threadIdx.x + bs]);
		}
		__syncthreads();
		if (threadIdx.x < bs) {
			b_min[threadIdx.x] = tmp_min;
			b_max[threadIdx.x] = tmp_max;
		}
		__syncthreads();
	}
	
	// get result
	if (threadIdx.x == 0)
	{
		atomicMin(result_min, b_min[0]);
		atomicMax(result_max, b_max[0]);
	}

}

int main()
{
	int *A;
	int *result_min;
	int *result_max;
	cudaMallocManaged((void**)&A, sizeof(int)*N);
	cudaMallocManaged((void**)&result_min, sizeof(int)*1);
	cudaMallocManaged((void**)&result_max, sizeof(int)*1);

	for (int i = 0; i < N; i++)
	{
		A[i] = rand() % 1024;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaEventQuery(start);

	int GSIZE = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	find_max_min_kernel <<< GSIZE,BLOCK_SIZE>>> (A,result_min,result_max,N,BLOCK_SIZE);

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "time is : " << time << " ms \n";

	cudaError err = cudaGetLastError();
	
	std::cout << "min is :" << result_min[0] << "\n";
	std::cout << "max is :" << result_max[0] << "\n";

	std::cout << "end\n";

}

