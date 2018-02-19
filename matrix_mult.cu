#include <stdio.h>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cstring>

#define N 16

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

inline void codeFlowControl(std::string s){
	std::cout << s << std::endl;
}

__global__ 
void matrixMulGPU( int *a, int *b, int *c,
                   int row_1, int col_1,
                   int row_2, int col_2)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y; 
    int total = 0;
	
    if(idx < row_1 && idy < col_2 && row_1 == col_2){
        total = 0;
        for(int runner = 0; runner < row_1; runner++){
            total += a[idx * row_1 + runner] * b[idy + runner * row_2];
        }
        c[idx * row_1 + idy] = total;
    }
}

__global__
void initMatrices(int *a, int *b, int *c, 
				  int row_a, int row_b, int row_c,
				  int col_a, int col_b, int col_c,
				  int val_a, int val_b, int val_c){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockIdx.y * blockDim.y;
	int idz = threadIdx.z + blockIdx.z * blockDim.z;
	
    if(idx < row_a * col_a && idy < row_b * col_b && idz < row_c * col_c){
		a[idx] = val_a;
		b[idy] = val_b;
		c[idz] = val_c;
	}
}

void matrixMulCPU(int *a, int *b, int *c){
	int sum = 0;
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
        	for (int k = 0; k < N; k++) {
          		sum += a[k + i * N] * b[j + k * N];
        	}
 			c[i*N + j] = sum;
        	sum = 0;
		}
	}
}

int main()
{
  /*************************************SQUARE_GPU_MALLOC****************************************************/
  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations
  int size   = N * N * sizeof (int); // Number of bytes of an N x N matrix

  checkCuda(cudaMallocManaged (&a, size));
  checkCuda(cudaMallocManaged (&b, size));
  checkCuda(cudaMallocManaged (&c_cpu, size));
  checkCuda(cudaMallocManaged (&c_gpu, size));

  /*Sequential Initialization*/
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = 2;//row;
      b[row*N + col] = 3;//col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }
 
  /**********************************************************************************************************/

  /************************************MAX_THREAD_COUNT_LEARNING*********************************************/ 	
  int device;
  cudaGetDevice(&device);
  struct cudaDeviceProp properties;
  cudaGetDeviceProperties(&properties, device);
  std::cout<<"using "<<properties.multiProcessorCount<<" multiprocessors"<<std::endl;
  std::cout<<"max threads per processor: "<<properties.maxThreadsPerMultiProcessor<<std::endl;
  /***********************************************************************************************************/
  
  /***********************************INITIALIZE_WITH_GPU*****************************************************/ 
  int row_a  = N,  col_a = N, row_b = N, col_b = N, row_c = N, col_c = N; //row col size of matrices
  int *a_gpu, *b_gpu, *c_gpu_2;
  int size_a = row_a * col_a * sizeof(int);
  int size_b = row_b * col_b * sizeof(int);
  int size_c = row_c * col_c * sizeof(int); 
  
  //Allocate memory for different size arrays
  size_t init_thread_size = N/2;
  size_t block_count_a = N / init_thread_size;//ceil((row_a * col_a + init_thread_size - 1) / init_thread_size);
  size_t block_count_b = N / init_thread_size; //ceil((row_b * col_b + init_thread_size - 1) / init_thread_size);
  size_t block_count_c =  N / init_thread_size;//ceil((row_c * col_c + init_thread_size - 1) / init_thread_size);

  checkCuda(cudaMallocManaged (&a_gpu, size_a));
  checkCuda(cudaMallocManaged (&b_gpu, size_b));
  checkCuda(cudaMallocManaged (&c_gpu_2, size_c));

  dim3 threads_per_block_init(init_thread_size, init_thread_size, init_thread_size);
  dim3 number_of_blocks_init(block_count_a, block_count_b, block_count_c);
  
  initMatrices<<<number_of_blocks_init, threads_per_block_init>>>(a_gpu, b_gpu, c_gpu_2, row_a, row_b, row_c, col_a, col_b, col_c, 2, 3, 0);
  checkCuda(cudaGetLastError());
  codeFlowControl("Stage-1");
  checkCuda(cudaDeviceSynchronize());
  codeFlowControl("Stage-2");
  /*****************************************************************************************************/

  size_t thread_in_block = N/2;
  size_t block_in_grid = N / thread_in_block;

  dim3 threads_per_block(thread_in_block, thread_in_block);
  dim3 number_of_blocks(block_in_grid, block_in_grid);

  matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a,b,c_gpu, N,N,N,N);
  //matrixMulGPU<<<number_of_blocks, threads_per_block>>>(a_gpu, b_gpu, c_gpu_2, row_a, col_a, row_b, col_b);
  
  codeFlowControl("Stage-3");
  checkCuda(cudaGetLastError());
  codeFlowControl("Stage-4");
  checkCuda(cudaDeviceSynchronize());
  codeFlowControl("END");

  // Call the CPU version to check our work
  //matrixMulCPU( a, b, c_cpu );
  matrixMulCPU(a, b, c_cpu);
  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d] value : %d expected : %d\n", row, col, c_gpu[row * N + col], c_cpu[row * N + col]);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  checkCuda(cudaFree(a));
  checkCuda(cudaFree(b));
  checkCuda(cudaFree( c_cpu ));
  checkCuda(cudaFree( c_gpu ));
}
