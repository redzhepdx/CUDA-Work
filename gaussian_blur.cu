#include<stdio.h>
#include<cmath>
#include "utils.h"
#define T_PI 3.0

//Gaussian Filter Kernel Creation
double* createGaussianKernel(int filterWidth){
	//Hyper-parameters for kernel
	double sigma = 10.0;
	double sum   = 0.0;
	double r, s  = 2.0 * sigma * sigma;
	
	//It will be 1D vector but size will equal to 2D version
	size_t size  = filterWidth * filterWidth * sizeof(double);
	
	//Allocate memory for kernel
	double *h_filter;
	checkCuda(cudaMallocManaged(&h_filter, size));//(double*)malloc(size * sizeof(double));

	//Non Normalized Kernel Creation
	for(int row = -(filterWidth/2); row <= filterWidth/2; row++){
		for(int col = -(filterWidth/2); col <= filterWidth/2; col++){
			r 		        = std::sqrt(row * row + col * col);
			int index       = (row + (int)filterWidth / 2) * filterWidth + (col + (int)filterWidth / 2);
			h_filter[index] = (std::exp(-(r * r) / s)) / (T_PI * s);
			sum			   += h_filter[index];
		}
	}
	
	//Kernel Normalization
	for(int row = 0; row < filterWidth; row++){
		for(int col = 0; col < filterWidth; col++){
			h_filter[row * filterWidth + col] /= sum;
		}
	}

	return h_filter;
}


__global__
void gaussianFilteringGPU(uchar *channel, uchar* res, double *filter, int rows, int cols, int filterWidth){
	int idx = threadIdx.x + blockIdx.x * blockDim.x;
	int idy = threadIdx.y + blockIdx.y * blockDim.y;
	
	if(idx < rows && idy < cols){
		int index = idx * cols + idy;
		
		if(idx == 0 || idy == 0 || idx == (rows - filterWidth/2) || idy == (cols - filterWidth/2)){
       		res[index] = channel[index];
		}
		else{
			int sum = 0;
			for(int filter_row = -(filterWidth/2); filter_row <= filterWidth/2; filter_row++){
				for(int filter_col   = -(filterWidth/2); filter_col <= filterWidth/2; filter_col++){
					int filter_index = (filter_row + (int)filterWidth / 2) * filterWidth + (filter_col + (int)filterWidth / 2);
					int img_index    = (idx + filter_row) * cols + (idy + filter_col);
					sum             += filter[filter_index] * channel[img_index];
        		}
			}
			//res[index] = (uchar) sum / (filterWidth * filterWidth);
			res[index] = (uchar) sum;
		}
	}
}


int main(){
	uchar *r_ch, *g_ch, *b_ch;
	uchar *res_r, *res_g, *res_b;
	
	std::string image_name   = "images/test.jpg";
	cv::Mat img              = cv::imread(image_name);
	cv::Mat res              = cv::Mat::zeros(img.size(), CV_8UC3);

	int rows                 = img.rows;
	int cols                 = img.cols;

	size_t ch_size           = rows * cols * sizeof(uchar);

	int numberOfThreads      = 32;
	int numberOfRowBlocks    = rows / numberOfThreads;
	int numberOfColBlocks    = cols / numberOfThreads;
	
	int filter_size          = 5;
	double * gaussian_kernel = createGaussianKernel(filter_size); 

	std::cout << "GENERATED KERNEL " << std::endl;
	for(int i = 0; i < filter_size; i++){
		for(int j = 0; j < filter_size; j++){
			std::cout << gaussian_kernel[i * filter_size + j] << " ";
		}
		std::cout << std::endl;
	}

	dim3 thread_count(numberOfThreads, numberOfThreads, 1);
	dim3 block_count(numberOfRowBlocks, numberOfColBlocks, 1);
	
	checkCuda(cudaMallocManaged(&r_ch, ch_size));
	checkCuda(cudaMallocManaged(&g_ch, ch_size));
	checkCuda(cudaMallocManaged(&b_ch, ch_size));
	checkCuda(cudaMallocManaged(&res_r, ch_size));
    checkCuda(cudaMallocManaged(&res_g, ch_size));
    checkCuda(cudaMallocManaged(&res_b, ch_size));
	
	//codeFlowControl("Debug_1");
	splitImage2Channels(img, r_ch, g_ch, b_ch);
	
	gaussianFilteringGPU<<<block_count, thread_count>>>(r_ch, res_r, gaussian_kernel, rows, cols, filter_size);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	//codeFlowControl("Debug_2");
	
	gaussianFilteringGPU<<<block_count, thread_count>>>(g_ch, res_g, gaussian_kernel, rows, cols, filter_size);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	//codeFlowControl("Debug_3");	

	gaussianFilteringGPU<<<block_count, thread_count>>>(b_ch, res_b, gaussian_kernel, rows, cols, filter_size);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	//codeFlowControl("Debug_4");

	vector2ImageRGB(res_r, res_g, res_b, res, rows, cols);

	cv::imwrite("blur_res.jpg", res);
	
	checkCuda(cudaFree(r_ch));
	checkCuda(cudaFree(g_ch));
	checkCuda(cudaFree(b_ch));
	checkCuda(cudaFree(res_r));
	checkCuda(cudaFree(res_g));
	checkCuda(cudaFree(res_b));
	checkCuda(cudaFree(gaussian_kernel));
}
