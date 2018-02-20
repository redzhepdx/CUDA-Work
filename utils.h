#ifndef _UTILS_H
#define _UTILS_H

#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <cstring>


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

#endif
