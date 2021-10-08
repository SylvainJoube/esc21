#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 0

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results,
                          const size_t n)
{
  // fill me
  __shared__ int sdata[10];
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

  // // Part 1 of 6: allocate the partial sums: How much space does it need?
  int* d_partial_sums_and_total;

  // // Part 1 of 6: launch one kernel to compute, per-block, a partial sum. How
  // much shared memory does it need?
  block_sum<<<num_blocks, block_size>>>(d_input, d_partial_sums_and_total,
                                        num_elements);

  // // Part 1 of 6: compute the sum of the partial sums
  block_sum<<<>>>();

  // // Part 1 of 6: copy the result back to the host
  int device_result = 0;

  std::cout << "Device sum: " << device_result << std::endl;

  // // Part 1 of 6: deallocate device memory

  return 0;
}
