/*Heat Conduction Computation Exercise From nvidia.qwiklab.com Optimized Version!!!
https://medium.com/@lucidlearning314/general-heat-conduction-equation-cartesian-coordinates-9be71b546b76
*/

#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <iostream>

#define I2D(num, c, r) ((r)*(num)+(c))

inline cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
  return result;
}

__global__
void step_kernel_mod(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;
  int idy = threadIdx.x + blockIdx.x * blockDim.x;
  int idx = threadIdx.y + blockIdx.y * blockDim.y;
  
  if (idy > 0 && idx > 0 && idy < nj-1 && idx < ni-1) {
    // find indices into linear memory
    // for central point and neighbours
    i00 = I2D(ni, idx, idy);
    im10 = I2D(ni, idx-1, idy);
    ip10 = I2D(ni, idx+1, idy);
    i0m1 = I2D(ni, idx, idy-1);
    i0p1 = I2D(ni, idx, idy+1);

    // evaluate derivatives
    d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
    d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

    // update temperatures
    temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
  }  
}

void initRandom(float *temp1_ref, float *temp2_ref, float *temp1, float *temp2, int ni, int nj){
  for( int i = 0; i < ni*nj; ++i) {
    temp1_ref[i] = temp2_ref[i] = temp1[i] = temp2[i] = (float)rand()/(float)(RAND_MAX/100.0f);
  }
}

void step_kernel_ref(int ni, int nj, float fact, float* temp_in, float* temp_out)
{
  int i00, im10, ip10, i0m1, i0p1;
  float d2tdx2, d2tdy2;


  // loop over all points in domain (except boundary)
  for ( int j=1; j < nj-1; j++ ) {
    for ( int i=1; i < ni-1; i++ ) {
      // find indices into linear memory
      // for central point and neighbours
      i00 = I2D(ni, i, j);
      im10 = I2D(ni, i-1, j);
      ip10 = I2D(ni, i+1, j);
      i0m1 = I2D(ni, i, j-1);
      i0p1 = I2D(ni, i, j+1);

      // evaluate derivatives
      d2tdx2 = temp_in[im10]-2*temp_in[i00]+temp_in[ip10];
      d2tdy2 = temp_in[i0m1]-2*temp_in[i00]+temp_in[i0p1];

      // update temperatures
      temp_out[i00] = temp_in[i00]+fact*(d2tdx2 + d2tdy2);
    }
  }
}

int main()
{
  int istep;
  int nstep = 200; // number of time steps
  int deviceId;
  int numberOfSMs;

  // Specify our 2D dimensions
  const int ni = 200;
  const int nj = 100;
  float tfac = 8.418e-5; // thermal diffusivity of silver

  float *temp1_ref, *temp2_ref, *temp1, *temp2, *temp_tmp;

  const int size = ni * nj * sizeof(float);
    
  cudaGetDevice(&deviceId);
  cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
  
  //CPU-GPU Memory Allocations
  checkCuda(cudaMallocManaged(&temp1_ref, size));
  checkCuda(cudaMallocManaged(&temp2_ref, size));
  checkCuda(cudaMallocManaged(&temp1, size));
  checkCuda(cudaMallocManaged(&temp2, size));
  
  //Avoid From CPU Page Faults
  cudaMemPrefetchAsync(temp1_ref, size, cudaCpuDeviceId);
  cudaMemPrefetchAsync(temp2_ref, size, cudaCpuDeviceId);
  cudaMemPrefetchAsync(temp1, size, cudaCpuDeviceId);
  cudaMemPrefetchAsync(temp2, size, cudaCpuDeviceId);
    
  initRandom(temp1_ref, temp2_ref, temp1, temp2, ni, nj);
 
  //CPU Computation
  for (istep=0; istep < nstep; istep++) {
    step_kernel_ref(ni, nj, tfac, temp1_ref, temp2_ref);

    // swap the temperature pointers
    temp_tmp = temp1_ref;
    temp1_ref = temp2_ref;
    temp2_ref= temp_tmp;
  }
  
  //Avoid From GPU Page Faults
  cudaMemPrefetchAsync(temp1_ref, size, deviceId);
  cudaMemPrefetchAsync(temp2_ref, size, deviceId);
  cudaMemPrefetchAsync(temp1, size, deviceId);
  cudaMemPrefetchAsync(temp2, size, deviceId);   
    
  int thread_size = 32;
  int scale = ni / nj;
  dim3 threads_per_block(thread_size, (int)(thread_size / scale), 1);
  dim3 number_of_blocks ((nj / threads_per_block.x) + 1, (ni / threads_per_block.y) + 1, 1);
    
  //GPU Computation
  for (int i=0; i < nstep; i++) {
      
      step_kernel_mod<<<number_of_blocks, threads_per_block>>>(ni, nj, tfac, temp1, temp2);
      //Check Errors and Synchronize 
      checkCuda(cudaGetLastError());
      checkCuda(cudaDeviceSynchronize());
      
      //Swap the temperature pointers
      temp_tmp = temp1;
      temp1 = temp2;
      temp2= temp_tmp;
  }

  float maxError = 0;
    
  // Output should always be stored in the temp1 and temp1_ref at this point
  for( int i = 0; i < ni*nj; ++i ) {
    if (abs(temp1[i]-temp1_ref[i]) > maxError) { maxError = abs(temp1[i]-temp1_ref[i]); }
  }

  // Check and see if our maxError is greater than an error bound
  if (maxError > 0.0005f)
    printf("Problem! The Max Error of %.5f is NOT within acceptable bounds.\n", maxError);
  else
    printf("The Max Error of %.5f is within acceptable bounds.\n", maxError);

  checkCuda(cudaFree( temp1_ref ));
  checkCuda(cudaFree( temp2_ref ));
  checkCuda(cudaFree( temp1 ));
  checkCuda(cudaFree( temp2 ));

  return 0;
}
