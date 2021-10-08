// includes, system
#include <cassert>
#include <iostream>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main()
{
  cudaSetDevice(MYDEVICE);
  // pointer and dimension for host memory
  int dimA = 8;
  std::vector<float> h_a(dimA); // x = dimA, y = 1, z = 1

  // pointers for device memory
  float *d_a, *d_b;

  // allocate and initialize host memory
  for (int i = 0; i < dimA; ++i) {
    h_a[i] = i;
  }

  // Part 1 of 5: allocate device memory
  size_t memSize = dimA * sizeof(float);
  cudaMalloc(&d_a, memSize);
  cudaMalloc(&d_b, memSize);

  // Part 2 of 5: host to device memory copy
  // Hint: the raw pointer to the underlying array of a vector
  // can be obtained by calling std::vector<T>::data()
  cudaMemcpy(d_a, h_a.data(), memSize, cudaMemcpyHostToDevice);

  // Part 3 of 5: device to device memory copy
  cudaMemcpy(d_b, d_a, memSize, cudaMemcpyDeviceToDevice);

  // clear host memory
  std::fill(h_a.begin(), h_a.end(), 0);

  // Part 4 of 5: device to host copy
  cudaMemcpy(h_a.data(), d_b, memSize, cudaMemcpyDeviceToHost);

  // Check for any CUDA errors
  checkCUDAError("cudaMemcpy calls");

  // verify the data on the host is correct
  for (int i = 0; i < dimA; ++i) {
    assert(h_a[i] == (float)i);
  }

  // Part 5 of 5: free device memory pointers d_a and d_b
  cudaFree(d_a);
  cudaFree(d_b);

  // Check for any CUDA errors
  checkCUDAError("cudaFree");

  // If the program makes it this far, then the results are correct and
  // there are no run-time errors.  Good work!
  std::cout << "Correct!" << std::endl;

  return 0;
}

void checkCUDAError(const char* msg)
{
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::cerr << "Cuda error: " << msg << " " << cudaGetErrorString(err)
              << std::endl;
    exit(-1);
  }
}
