#include <stdio.h>

int main()
{
  int deviceId;
  
  cudaGetDevice(&deviceId);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, deviceId);
    
  int computeCapabilityMajor = props.major;
  int computeCapabilityMinor = props.minor;
  int multiProcessorCount    = props.multiProcessorCount;
  int warpSize               = props.warpSize;
  std::cout << "Device ID : " << deviceId
            << "\nNumber of SMs : " << multiProcessorCount 
            << "\nCompute Capability Major : " << computeCapabilityMajor 
            << "\nCompute Capability Minor : " << computeCapabilityMinor 
            << "\n Warp Size : " << warpSize << std::endl;
            
  return 0;
}
