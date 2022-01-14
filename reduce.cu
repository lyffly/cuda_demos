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


__global__ void reduce_kernel(int *A,int *result_sum, int n, int bsize)
{
	__shared__ int s_sum[BLOCK_SIZE];
	
	//copy data to share memory
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	s_sum[threadIdx.x] = 0;
	
	if (index < n) {
		s_sum[threadIdx.x] = A[index];
	}
	__syncthreads();

	// find min max loop
	for (int bs = BLOCK_SIZE / 2; bs > 0; bs /= 2)
	{
		int tmp = 0;
		
		if (threadIdx.x < bs) {
			tmp = s_sum[threadIdx.x] + s_sum[threadIdx.x + bs];
		}
		__syncthreads();
		if (threadIdx.x < bs) {
			s_sum[threadIdx.x] = tmp;
			
		}
		__syncthreads();
	}
	
	// get result
	if (threadIdx.x == 0)
	{
		atomicAdd(result_sum, s_sum[0]);
		
	}

}

int main()
{
	int *A;
	int *result_sum;
	
	cudaMallocManaged((void**)&A, sizeof(int)*N);
	cudaMallocManaged((void**)&result_sum, sizeof(int)*1);
	

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
	reduce_kernel <<< GSIZE,BLOCK_SIZE>>> (A, result_sum,N,BLOCK_SIZE);

	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	std::cout << "time is : " << time << " ms \n";

	cudaError err = cudaGetLastError();
	std::cout << " Error information :"<< cudaGetErrorString(err)  << "\n";

	
	std::cout << "sum result is :" << result_sum[0] << "\n";
	
	cudaFree(A);
	cudaFree(result_sum);
	std::cout << "end\n";

}

