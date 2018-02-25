#include <stdio.h>
#include <assert.h>
#include <iostream>

#define N 2048 * 2048 // Number of elements in each vector

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

__global__ void saxpy(int * a, int * b, int * c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    for(int i = tid; i < N; i+=stride){
        c[i] = 2 * a[i] + b[i];
    }
}


void saxpy_s(int * a, int * b, int * c)
{
    for(int i = 0; i < N; i++){
        c[i] = 2 * a[i] + b[i];
    }
}

void init_vector(int value, int *a){
    for(int i = 0; i < N; i++){
           a[i] = value;
    }
}

int main()
{
    int *a, *b, *c;

    int size = N * sizeof (int); // The total number of bytes per vector
    
    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);
    
    cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);
    cudaMemPrefetchAsync(b, size, cudaCpuDeviceId);
    cudaMemPrefetchAsync(c, size, cudaCpuDeviceId);
    
    init_vector(2, a);
    init_vector(1, b);
    init_vector(0, c);
    
    cudaMemPrefetchAsync(a, size, deviceId);
    cudaMemPrefetchAsync(b, size, deviceId);
    cudaMemPrefetchAsync(c, size, deviceId);
    
    int threads_per_block = 256;
    int number_of_blocks  = (N + threads_per_block - 1) / threads_per_block;
    //std::cout << number_of_blocks << std::endl;

    saxpy <<< number_of_blocks, threads_per_block >>> ( a, b, c );
    
    checkCuda(cudaGetLastError());
    checkCuda(cudaDeviceSynchronize());
    //saxpy_s(a,b,c);
    
    // Print out the first and last 5 values of c for a quality check
    for( int i = 0; i < 5; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");
    for( int i = N-5; i < N; ++i )
        printf("c[%d] = %d, ", i, c[i]);
    printf ("\n");

    checkCuda(cudaFree( a ));
    checkCuda(cudaFree( b ));
    checkCuda(cudaFree( c ));
}
