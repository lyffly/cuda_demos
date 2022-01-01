#include <iostream>
#include <time.h>
#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"


using namespace std;


__global__ void vector_add_kernel(const double* d_x, const double* d_y, double * d_out, unsigned int length)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;	
	if (idx < length)
	{
		d_out[idx] = d_x[idx] + d_y[idx];		
	}
}


int main()
{
	const unsigned int length = 400000;
	double h_x[length];
	double h_y[length];	
	double h_out[length];
	
	for (int i = 0; i < length; i++)
	{
		h_x[i] = 1.1111;
		h_y[i] = 2.2222;
		h_out[i] = 0.0;
	}

	double *d_x;
	double *d_y;
	double *d_out;

	cudaMalloc((void**)&d_x, length * sizeof(double));
	cudaMalloc((void**)&d_y, length * sizeof(double));
	cudaMalloc((void**)&d_out, length * sizeof(double));
	
	cudaMemcpy(d_x, h_x, length * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, h_y, length * sizeof(double), cudaMemcpyHostToDevice);

	dim3 blocksize(1024);
	dim3 gridsize((length + blocksize.x-1)/ blocksize.x);

	//start record time
	clock_t t1,t2;
	t1 = clock();

	vector_add_kernel<<<gridsize, blocksize >>> (d_x,d_y, d_out, length);
	cudaDeviceSynchronize();

	//end record time
	t2 = clock();
	std::cout << (double)(t2-t1)/CLOCKS_PER_SEC*1000.0 << std::endl;

	cudaMemcpy(h_out, d_out, length * sizeof(double), cudaMemcpyDeviceToHost);

	cudaFree(d_x);
	cudaFree(d_y);
	cudaFree(d_out);

	std::cout << h_out[length-1] << std::endl;
	char s;
	std::cin >> s;
}







