#include <stdio.h>
#include "utils.h"

__global__
void rgb2GrayTransform(uchar *res, uchar *r_ch, uchar *g_ch, uchar *b_ch, int rows, int cols){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int index = idx * cols + idy;
	if(idx < rows && idy < cols){
		res[index] = (r_ch[index] + g_ch[index] + b_ch[index]) / 3;
	}
}

int main( int argc, char** argv ){

	uchar *r_ch, *g_ch, *b_ch, *res;

	std::string image_name = "test.jpg";
	cv::Mat img 		   = cv::imread(image_name);
	int rows               = img.rows;
	int cols			   = img.cols;
	size_t ch_size         = rows * cols * sizeof(uchar);
	int numberOfThreads    = 32;
	int numberOfRowBlocks  = rows / numberOfThreads;
	int numberOfColBlocks  = cols / numberOfThreads;

	dim3 thread_count(numberOfThreads, numberOfThreads);
	dim3 block_count(numberOfRowBlocks, numberOfColBlocks);

	checkCuda(cudaMallocManaged(&r_ch, ch_size));
	checkCuda(cudaMallocManaged(&g_ch, ch_size));
	checkCuda(cudaMallocManaged(&b_ch, ch_size));
	checkCuda(cudaMallocManaged(&res, ch_size));

    splitImage2Channels(img, r_ch, g_ch, b_ch);

	rgb2GrayTransform<<<block_count, thread_count>>>(res, r_ch, g_ch, b_ch, rows, cols);

	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());

	cv::Mat result(rows, cols, CV_8UC1, cv::Scalar(0));
	
	vector2Image(res, result, rows, cols);

	cv::imwrite("result.jpg", result);

	return 0;
}
