
#include <iostream>


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

const int N = 10000;



__global__ void transpose1(const double *A,double *B,const int N)
{
	int col = blockIdx.x *blockDim.x + threadIdx.x;
	int row = blockIdx.y *blockDim.y + threadIdx.y;
	
	if (row < N && col < N)
	{
		B[col*N + row] = A[row *N + col];
		//B[row *N + col] = A[col * N + row];
	}
}

int main()
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	cudaEventQuery(start);

	double  *host_A;
	double  *host_B;

	cudaMallocHost((void**)&host_A, N*N * sizeof(double));
	cudaMallocHost((void**)&host_B, N*N * sizeof(double));


	double *d_A;
	double *d_B;
	cudaMalloc((void**)&d_A, N*N * sizeof(double));
	cudaMalloc((void**)&d_B, N*N * sizeof(double));
    	
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			host_A[i*N+j] = 1.2 + i +j;
		}
	}	
	
	//cudaMemcpyToSymbol(d_input_M, host_input_M, N*N*sizeof(int));
	cudaMemcpy(d_A, host_A, N*N * sizeof(double),cudaMemcpyHostToDevice);

	int ksize = 32;
	dim3 blocksize(ksize, ksize);
	int gsize = (int)((N + ksize - 1) / ksize);
	dim3 gridsize(gsize,gsize);
	
	transpose1 <<<gridsize, blocksize>>>(d_A,d_B,N);


	cudaMemcpy(host_B, d_B, N*N * sizeof(double), cudaMemcpyDeviceToHost);
	//cudaMemcpyFromSymbol(host_output_M, d_output_M, N*N*sizeof(int));


	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);
	std::cout <<"time is : "<< time <<  " ms \n";


	cudaError err =  cudaGetLastError();
	
	std::cout<< host_A[5*N+2] << "\n";
	std::cout << host_B[2*N+5] << "\n";

	std::cout << host_A[77 * N + 22] << "\n";
	std::cout << host_B[22 * N + 77] << "\n";

	std::cout << "end\n";

	char s;
	std::cin >> s;

}

