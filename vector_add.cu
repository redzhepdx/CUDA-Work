#include <stdio.h>
#include <assert.h>

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}


__global__
void initVectorGpu(float *a, float value, int N){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        a[idx] = value;
    }
}


__global__
void addVectorsGpu(float *result, float *a, float *b, int N){
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int stride = blockDim.x * gridDim.x;

  for(int i = idx; i < N; i += stride){
    result[i] = a[i] + b[i];
  }
}

void checkElementsAre(float target, float *array, int N)
{
  for(int i = 0; i < N; i++)
  {
    if(array[i] != target)
    {
      printf("FAIL: array[%d] - %0.0f does not equal %0.0f\n", i, array[i], target);
      exit(1);
    }
  }
  printf("SUCCESS! All values added correctly.\n");
}

int main()
{
  const int N = 2<<20;
  size_t size = N * sizeof(float);

  float *a;
  float *b;
  float *c;

  a = (float *)malloc(size);
  b = (float *)malloc(size);
  c = (float *)malloc(size);
    
  checkCuda(cudaMallocManaged(&a, size));
  checkCuda(cudaMallocManaged(&b, size));
  checkCuda(cudaMallocManaged(&c, size));
  
  size_t threadsPerBlock;
  size_t numberOfBlocks;

  threadsPerBlock = 256;
  numberOfBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
  printf("Threads : %ld Blocks : %ld", threadsPerBlock, numberOfBlocks);
    
  initVectorGpu<<<numberOfBlocks,threadsPerBlock>>>(a, 3, N);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
    
  initVectorGpu<<<numberOfBlocks,threadsPerBlock>>>(b, 4, N);
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  
  initVectorGpu<<<numberOfBlocks,threadsPerBlock>>>(c, 0, N);  
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());  
  
  addVectorsGpu<<<numberOfBlocks, threadsPerBlock>>>(c, a, b, N);
  
  checkCuda(cudaGetLastError());
  checkCuda(cudaDeviceSynchronize());
  checkElementsAre(7, c, N);
    
  checkCuda( cudaFree(a) );
  checkCuda( cudaFree(b) );
  checkCuda( cudaFree(c) );
}
