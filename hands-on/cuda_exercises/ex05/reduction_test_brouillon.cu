#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 1

constexpr int num_elements = 1 << 18;
constexpr uint num_blocks = num_elements >> 10; // div by 1024
constexpr uint block_size = num_elements / num_blocks;

// constexpr uint num_blocks = 1 << 8; //num_elements >> 10; // div by 1024
// constexpr uint block_size = 1 << 10; // num_elements / num_blocks;

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results,
                          const size_t n, const int block_sizeee)
{
  // fill me
  // shared memory : chunk of size block_size from the input
  __shared__ int sdata[block_size];
  uint block_id = blockIdx.x;
  uint thread_id = threadIdx.x;
  // fill the shared memory :
  // each thread of a block fills its cell
  sdata[thread_id] = input[block_size * block_id + thread_id];

  // Wait for the shared memory to be full
  __syncthreads();
  // One single thread sums all the elements of the block
  if (thread_id == 0) {
    int psum = 0;
    for (uint i = 0; i < block_size; ++i) {
      psum += 1;//sdata[i];
    }
    per_block_results[block_id] = psum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(void)
{
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<> distrib(-10, 10);
  // create array of 256ki elements
  const int num_elements = 1 << 18;
  // generate random input on the host
  std::vector<int> h_input(num_elements);
  for (auto& elt : h_input) {
    elt = distrib(gen);
  }

  const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);
  std::cerr << "Host sum: " << host_result << std::endl;

  // //Part 1 of 6: move input to device memory
  int* d_input;

  // all the elements to sum
  uint in_size = num_elements * sizeof(int);

  // partial sums
  uint num_blocks = num_elements >> 10; // div by 1024
  const uint block_size = num_elements / num_blocks;
  // partial sum array
  const uint out_psm_size = num_blocks * sizeof(int);

  // Alloc and copy input data
  cudaMalloc(&d_input, in_size);
  cudaMemcpy(d_input, h_input.data(), in_size, cudaMemcpyHostToDevice);

  // // Part 1 of 6: allocate the partial sums: How much space does it need?
  int* d_partial_sums_and_total;
  cudaMalloc(&d_partial_sums_and_total, out_psm_size);

  int* d_result;
  cudaMalloc(&d_result, sizeof(int));


  // // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How
  // much shared memory does it need?
  block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total,
                                        num_elements, block_size);

  int h_partial_sums_and_total[num_blocks];
  cudaMemcpy(&h_partial_sums_and_total, d_partial_sums_and_total, out_psm_size, cudaMemcpyDeviceToHost);
  for (uint ib = 0; ib < num_blocks; ++ib) {
    std::cout << "b(" << ib << ") = " << h_partial_sums_and_total[ib] << std::endl;
  }

  // // Part 1 of 6: compute the sum of the partial sums
  block_sum<<<1, num_blocks>>>(d_partial_sums_and_total, d_result, 0, num_blocks);

  // // Part 1 of 6: copy the result back to the host
  int device_result = 0;
  cudaMemcpy(&device_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);


  std::cout << "Device sum: " << device_result << std::endl;

  // // Part 1 of 6: deallocate device memory

  return 0;
}
