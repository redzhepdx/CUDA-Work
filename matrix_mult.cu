#include <stdio.h>

#define N  64

__global__ 
void matrixMulGPU( int *a, int *b, int *c,
                   int row_1, int col_1,
                   int row_2, int col_2)
{
    //thread = 0-64
    //block = 0 - n-thread+1 / thread
    //dim = # of thread in block
    //grid = # of block in grid
    
    int total = 0;
    
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y; 
    
    if(idx < row_1 && idy < col_2 && row_1 == col_2){
        total = 0;
        for(int runner = 0; runner < row_1; runner++){
            total += a[idx * row_1 + runner] * b[idy + runner * row_2];
        }
        c[idx + row_1 * idy] = total;
    }
    
}

__global__
void initMatrices(int *a, int *b, int *c){

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y; 

    int stride_x = blockDim.x * gridDim.x;
    int stride_y = blockDim.y * gridDim.y;

    for(int i = idx; i < N; i)
    
}

int main()
{
  int *a, *b, *c_cpu, *c_gpu; // Allocate a solution matrix for both the CPU and the GPU operations

  int size = N * N * sizeof (int); // Number of bytes of an N x N matrix

  // Allocate memory
  cudaMallocManaged (&a, size);
  cudaMallocManaged (&b, size);
  cudaMallocManaged (&c_cpu, size);
  cudaMallocManaged (&c_gpu, size);

  // Initialize memory; create 2D matrices
  for( int row = 0; row < N; ++row )
    for( int col = 0; col < N; ++col )
    {
      a[row*N + col] = row;
      b[row*N + col] = col+2;
      c_cpu[row*N + col] = 0;
      c_gpu[row*N + col] = 0;
    }

  /*
   * Assign `threads_per_block` and `number_of_blocks` 2D values
   * that can be used in matrixMulGPU above.
   */
  size_t thread_in_block = 256;
  size_t block_in_grid = (N+thread_in_block-1)/thread_in_block;
    
  dim3 threads_per_block(thread_in_block, thread_in_block, 1);
  dim3 number_of_blocks(block_in_grid, block_in_grid, 1);

  matrixMulGPU <<< number_of_blocks, threads_per_block >>> ( a, b, c_gpu, N, N, N, N);

  cudaDeviceSynchronize();

  // Call the CPU version to check our work
  //matrixMulCPU( a, b, c_cpu );

  // Compare the two answers to make sure they are equal
  bool error = false;
  for( int row = 0; row < N && !error; ++row )
    for( int col = 0; col < N && !error; ++col )
      if (c_cpu[row * N + col] != c_gpu[row * N + col])
      {
        printf("FOUND ERROR at c[%d][%d]\n", row, col);
        error = true;
        break;
      }
  if (!error)
    printf("Success!\n");

  // Free all our allocated memory
  cudaFree(a); cudaFree(b);
  cudaFree( c_cpu ); cudaFree( c_gpu );
}
