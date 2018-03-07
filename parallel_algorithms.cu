#include "utils.h"
#include <iostream>

__global__
void shmem_gpu_reduce(double* input, double* output){
	extern __shared__ double sdata[];	

	int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int t_idx = threadIdx.x;

	//Load shared mem from global mem
	sdata[t_idx] = input[g_idx];
	__syncthreads();
	
	//reduction in shared mem
	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
		if(t_idx < stride){
			sdata[t_idx] += sdata[t_idx + stride];
		}
		__syncthreads();
	}

	if(t_idx == 0){
		output[blockIdx.x] = sdata[t_idx];
	}
}

__global__
void gpu_reduce(double* input, double* output){

	int g_idx = threadIdx.x + blockIdx.x * blockDim.x;
	int t_idx = threadIdx.x;

	for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
		if(t_idx < stride){
			input[g_idx] += input[g_idx + stride];
			//printf("indexes : %d - %d", g_idx, g_idx + stride);
		}
		__syncthreads();
	}

	if(t_idx == 0){
		output[blockIdx.x] = input[g_idx];
	}
}

void reduce_algorithm(){
	double* list_in;
	double* list_out;
	bool useShared   = true;
	int capacity     = 16;
	size_t size      = capacity * sizeof(double);
	
	int thread_count = capacity;//(int)std::sqrt(capacity);
	int block_count  = 1;//(int)std::sqrt(capacity);

	checkCuda(cudaMallocManaged(&list_in, size));
 
	checkCuda(cudaMallocManaged(&list_out, size));
	
	for(int i = 0; i < capacity; i++){
		list_in[i]  = i;
		list_out[i] = 0;
	}
	
	if(!useShared){
		gpu_reduce<<<block_count, thread_count>>>(list_in, list_out);
		checkCuda(cudaGetLastError());
    	checkCuda(cudaDeviceSynchronize());
	}
	else{
		std::cout << "Using Shared Memory" << std::endl;
		//Using Shared Memory <<< block count, thread count, shared memory size
		shmem_gpu_reduce<<<block_count, thread_count, thread_count * sizeof(double)>>>(list_in, list_out);
		checkCuda(cudaGetLastError());
		checkCuda(cudaDeviceSynchronize());
	}

	for(int i = 0; i < capacity; i++){
        std::cout << list_out[i] << " ";
    }

	std::cout << std::endl;

	checkCuda(cudaFree(list_in));
	checkCuda(cudaFree(list_out));
}


#ifndef BIN_COUNT
#define BIN_COUNT 16
#endif

__global__
void naive_histo(int* input, int* bins, int size){
	int idx    = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	
	for (int index = idx; index < size; index += stride){
		int bin = input[index] % BIN_COUNT;
		//using atomic add because we don't want memory access collision, it slow downs computation little bit but giving synchronized memory access and increment chance
		atomicAdd(&(bins[bin]), 1);
	}
}



void compute_histogram(){
	int *input, *bins;

	int size         = 1 << 16;
	int thread_count = 1024;
	int block_count  = (size + thread_count - 1) / thread_count;
	
	std::cout << size << std::endl;

	checkCuda(cudaMallocManaged(&input, size*sizeof(int)));
	checkCuda(cudaMallocManaged(&bins, BIN_COUNT*sizeof(int)));

	for(int i = 0; i < size; i++){
		input[i] = i;
	}

	naive_histo<<<block_count, thread_count>>>(input, bins, size);
	checkCuda(cudaGetLastError());
	checkCuda(cudaDeviceSynchronize());
	
	for(int i = 0; i < BIN_COUNT; i++){
		std::cout << bins[i] << " ";
	}
	std::cout << std::endl;

	checkCuda(cudaFree(input));
	checkCuda(cudaFree(bins));
}



void serial_scan_exclusive(){
	const int size = 10;
	int in[]       = {1,2,3,4,5,6,7,8,9,10};
	int acc        = 0;
	int out[size];

	for(int i = 0; i < size; i++){
		out[i] = acc;
		acc   += in[i];
	}

	for(int i = 0; i < size; i++){
		std::cout << out[i] << " ";
	}
	std::cout << std::endl;
}



int main(){
	reduce_algorithm();
	//compute_histogram();
	return 0;
}
