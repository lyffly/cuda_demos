
#include <iostream>

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;


__global__ void rgb2gray_kernel(const uchar3* d_in,unsigned char * d_out,uint imgheight,uint imgwidth)
{
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int idy = blockIdx.y * blockDim.y + threadIdx.y;
	if (idx < imgwidth && idy < imgheight)
	{
		uchar3 rgb = d_in[idy*imgwidth + idx];
		unsigned char gray = 0.299f*rgb.x + 0.587f*rgb.y + 0.114f*rgb.z;

		d_out[idy*imgwidth + idx] = gray;
	}
}


int main()
{
	Mat rgb_img = imread("demo.png", 1);

	const uint img_height = rgb_img.rows;
	const uint img_width = rgb_img.cols;

	Mat gray_img(img_height, img_width, CV_8UC1, Scalar(0));

	uchar3 *d_in;
	unsigned char *d_out;

	cudaMalloc((void**)&d_in, img_height*img_width * sizeof(uchar3));
	cudaMalloc((void**)&d_out, img_height*img_width * sizeof(unsigned char));
	
	cudaMemcpy(d_in, rgb_img.data, img_height*img_width * sizeof(uchar3), cudaMemcpyHostToDevice);
		
	dim3 blocksize(32, 32);
	int grid_x = (img_width + blocksize.x - 1) / blocksize.x;
	int grid_y = (img_height + blocksize.y - 1) / blocksize.y;
	dim3 gridsize(grid_x, grid_y);

	rgb2gray_kernel <<<gridsize, blocksize >>> (d_in, d_out, img_height, img_width);
	
	cudaMemcpy(gray_img.data, d_out, img_height*img_width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

	cudaFree(d_in);
	cudaFree(d_out);

	cv::imshow("gray image result", gray_img);
	
	cv::waitKey(0);
	//char s;
	//std::cin >> s;

	
}

