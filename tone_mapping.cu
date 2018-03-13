#include "utils.h"
#include <stdio.h>
#include <cmath>
#include <stdint.h>
#include <float.h>
#include <string>
#include <sstream>
#include <iostream>

#ifndef MIN
#define MIN(X,Y) (((X)>(Y))?(Y):(X))
#endif


#ifndef MAX
#define MAX(X,Y) (((X)>(Y))?(X):(Y))
#endif

#ifndef GRAY_LEVEL
#define GRAY_LEVEL 256
#endif 

/*
HDR(High Dynamic Range) Algorithm

Histogram Equalization

-Histogram Computation of Brightness 
-Scan of the histogtram
-Min&Max and Brightness
*/

__global__
void find_min_max_gpu(double* logLuminance, double *out, size_t size, int minmax){
	extern __shared__ double shared[];

	int global_id = threadIdx.x + blockIdx.x * blockDim.x;
	int thread_id = threadIdx.x;

	if (global_id < size){
		shared[thread_id] = logLuminance[global_id];
	}
	else{
		if(!minmax){
			shared[thread_id] = FLT_MAX;
		}
		else{
			shared[thread_id] = -FLT_MAX;
		}
	}

	//Wait for all threads to copy the memory
	__syncthreads();


	for(int s = blockDim.x / 2; s>0; s /= 2){
		if(thread_id < s){
			if(!minmax){
				shared[thread_id] = MIN(shared[thread_id], shared[thread_id + s]);
			}
			else{
				shared[thread_id] = MAX(shared[thread_id], shared[thread_id + s]);
			}
		}
		__syncthreads();
	}

	if(thread_id == 0){
		out[blockIdx.x] = shared[0];
	}
}

__global__
void cdf_to_image(double *input, int *bins, int size){
	int idx    = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	for(int index = idx; index < size; index+=stride){
		input[index] = (double)(bins[(int)input[index]] - 1) * (GRAY_LEVEL - 1) / size;
	}
}

void find_max_min_cpu(double* logLuminance, double &maxLum, double &minLum, int size){
	maxLum = logLuminance[0];
	minLum = logLuminance[0];

	for(int i = 0; i < size; i++){
		maxLum = MAX(logLuminance[i], maxLum);
		minLum = MIN(logLuminance[i], minLum);
	}
	std::cout << "Max : " << maxLum << " Min : " << minLum << std::endl;
}

__global__
void compute_histogram(double *input, int *bins ,
					   double min, double range,
					   int bin_count, size_t size){
	int idx    = threadIdx.x + blockIdx.x * blockDim.x;
	//int stride = blockDim.x * gridDim.x;
	
	if(idx < size){
		int bin_index = (int)(((input[idx] - min) / range) * bin_count);
		//printf("%d\n", bin_index);
		atomicAdd(&(bins[bin_index]), 1);
	}
}

//Parallel Cumulative Distribution Computation with Hills Steele Scan Algorithm
__global__
void cumulative_distribution(int *bins, int bin_size){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	
	if(idx >= bin_size){
		return;
	}

	for(int s = 1; s <= bin_size; s*=2){

		int spot = idx - s;
		int val = 0;

		if (spot >= 0){
			val = bins[spot];
		}
		__syncthreads();

		if (spot >= 0){
			bins[idx] += val;
		}
		__syncthreads();
	}
	
}

double reduce_minmax(double *input, size_t size, int minmax){
	double *maxLum, *temp_ch;

	int BLOCK_SIZE      = 16;
	size_t curr_size    = size;
	int numberOfThreads = BLOCK_SIZE;
	int shared_mem_size = numberOfThreads * sizeof(double);

	checkCuda(cudaMallocManaged(&temp_ch, size * sizeof(double)));
	temp_ch = input;

	while(1){
       
		checkCuda(cudaMallocManaged(&maxLum, curr_size * sizeof(double)));	
		int numberOfBlocks  = getMaxSize(curr_size, BLOCK_SIZE);
		std::cout << "BLock Count : " << numberOfBlocks << " size : " << curr_size << std::endl;
		//codeFlowControl(std::to_string(numberOfBlocks));
			
		find_min_max_gpu <<< numberOfBlocks, numberOfThreads, shared_mem_size >>> (temp_ch, maxLum, curr_size, minmax);
		checkCuda(cudaGetLastError());
		checkCuda(cudaDeviceSynchronize());

		codeFlowControl("Debug Loop");
		
		//Free Memory
		checkCuda(cudaFree(temp_ch));

		temp_ch = maxLum;

		if (curr_size < BLOCK_SIZE){
			break;
		}
		
		//Reset New Memory Size
		curr_size = getMaxSize(curr_size, BLOCK_SIZE);
    }

	double res = maxLum[0];
	checkCuda(cudaFree(maxLum));
	return res;
}


void apply_tone_mapping(){
	uchar *r_ch, *g_ch, *b_ch;

    double *y_ch, *u_ch, *v_ch;
	int *bins, *d_cdf;

	cv::Mat yuv;

	std::string image_name = "images/tone_map2.jpg";
	cv::Mat	img            = cv::imread(image_name);
	cv::Mat res_yuv        = cv::Mat::zeros(img.size(), CV_8UC3);

	int rows               = img.rows;
	int cols               = img.cols;
	int bin_count          = 256;
	int bin_size           = bin_count * sizeof(int);
	size_t size            = rows * cols;
	size_t ch_size_rgb     = size * sizeof(uchar);
	size_t ch_size_yuv     = size * sizeof(double);

	/************THREAD BLOCK SIZE CONFIGURATION***************/
	int numberOfThreads    = 1024;
	int numberOfBlocks	   = getMaxSize(size, numberOfThreads);//(int)std::ceil((float)size / (float)numberOfThreads) + 1;
	int numberOfRowBlocks  = rows / numberOfThreads;
	int numberOfColBlocks  = cols / numberOfThreads;
	dim3 thread_count(numberOfThreads, numberOfThreads);
	dim3 block_count(numberOfRowBlocks, numberOfColBlocks);
	
	/****************MEMORY CONVERSION PART*******************/
	checkCuda(cudaMallocManaged(&r_ch, ch_size_rgb));
	checkCuda(cudaMallocManaged(&g_ch, ch_size_rgb));
	checkCuda(cudaMallocManaged(&b_ch, ch_size_rgb));
	checkCuda(cudaMallocManaged(&y_ch, ch_size_yuv));
	checkCuda(cudaMallocManaged(&u_ch, ch_size_yuv));
	checkCuda(cudaMallocManaged(&v_ch, ch_size_yuv));
	checkCuda(cudaMallocManaged(&bins, bin_size));	
	checkCuda(cudaMallocManaged(&d_cdf, bin_size));

	//RGB -> YUV
	cv::cvtColor(img, yuv, CV_BGR2YUV);
	
	//Split image to seperate channels
	splitImage2ChannelsDouble(yuv, v_ch, u_ch, y_ch);

	/********************MIN_MAX_REDUCE*********************/
	//TODO Contains Bug , CPU version for Now
	//double max = reduce_minmax(y_ch, size, 1);
	//double min = reduce_minmax(y_ch, size, 0);
	/*******************************************************/

	double min, max;
	find_max_min_cpu(y_ch, max, min, size);
	
	double range = max - min;

	std::cout << "MAX : " << max << "MIN : " << min << std::endl;
	fillVector(bins, 0, bin_size);	

	compute_histogram <<< numberOfBlocks, numberOfThreads >>> (y_ch, bins, min, range, bin_count, size);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	
	/*
	for(int i = 0; i < bin_count; i++){
		std::cout << "BİN_" <<  i << " : " << bins[i] << std::endl;
	}	
	*/

	int binBlockCount = getMaxSize(bin_count, numberOfThreads);

	cumulative_distribution <<< binBlockCount, numberOfThreads >>>(bins, bin_count);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	
	/*
	for(int i = 0; i < bin_count; i++){
		std::cout << "CDF_" << i << " : " << bins[i] << std::endl;
	}
	*/

	cdf_to_image<<< numberOfBlocks, numberOfThreads >>>	(y_ch, bins, size);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	vector2ImageRGB(v_ch, u_ch, y_ch, res_yuv, rows, cols);
	
	/*
	for(int i = 0; i < 200; i++)
		std::cout << "Color : " <<(int)(uchar)y_ch[i] << std::endl; 
	*/

	cv::cvtColor(res_yuv, res_yuv, CV_YUV2BGR);
	cv::imwrite("tone_map.jpg", res_yuv);

	checkCuda(cudaFree(r_ch));
	checkCuda(cudaFree(g_ch));
	checkCuda(cudaFree(b_ch));
	checkCuda(cudaFree(y_ch));
	checkCuda(cudaFree(u_ch));
	checkCuda(cudaFree(v_ch));
	checkCuda(cudaFree(bins));
}


int main(){
	apply_tone_mapping();

	return 0;
}
