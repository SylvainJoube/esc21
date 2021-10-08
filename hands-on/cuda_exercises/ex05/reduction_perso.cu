#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Here you can set the device ID that was assigned to you
#define MYDEVICE 1
#define BLOCK_SIZE 512
#define BLOCKS_NUMBER 512

// Part 1 of 6: implement the kernel
__global__ void block_sum(const int* input, int* per_block_results,
                          const size_t n)
{
  // fill me
  __shared__ int shared_mem[BLOCK_SIZE];
  uint block_id = blockIdx.x;
  uint thread_id = threadIdx.x;

  shared_mem[thread_id] = input[block_id * BLOCK_SIZE + thread_id];

  __syncthreads();
  
  /*if (thread_id == 0) {
    int sum = 0;
    for (uint i = 0; i < BLOCK_SIZE; ++i) {
      sum += shared_mem[i];
    }
    per_block_results[block_id] = sum;
  }*/

  uint max_allowed_id = BLOCK_SIZE / 2;

  while (thread_id < max_allowed_id) {
    shared_mem[thread_id] += shared_mem[thread_id + max_allowed_id];
    max_allowed_id = max_allowed_id / 2;
    __syncthreads();
  }

  
  
  if (thread_id == 0) {
    per_block_results[block_id] = shared_mem[0];
  }
}

void cpu_block_sum(int block_id, int thread_id, int* global_inoutput, int* per_block_results) {
  // BLOCK_SIZE is a const si no : const size_t n
  
  uint istart = block_id * BLOCK_SIZE;
  
  /*if (thread_id == 0) {
    int sum = 0;
    for (uint i = 0; i < BLOCK_SIZE; ++i) {
      sum += shared_mem[i];
    }
    per_block_results[block_id] = sum;
  }*/

  uint max_allowed_id = BLOCK_SIZE / 2;

  while (thread_id < max_allowed_id) {
    global_inoutput[istart + thread_id] += global_inoutput[istart + thread_id + max_allowed_id];
    max_allowed_id = max_allowed_id / 2;
  }
  
  if (thread_id == 0) {
    per_block_results[block_id] = global_inoutput[istart + thread_id];
  }
}

void cpu_reduction(const int* input, int* per_block_results, const size_t input_size, uint blocks_number) {
  int global_inoutput[input_size];
  // Copy the input to a teporary array
  for (uint i = 0; i < input_size; ++i) {
    global_inoutput[i] = input[i];
  }
  // For each block
  for (uint ib = 0; ib < blocks_number; ++ib) {
    // For each thread, descending order
    for (int ith = BLOCK_SIZE - 1; ith >= 0; --ith) {
    //for (int ith = 0; ith < BLOCK_SIZE; ++ith) {
      cpu_block_sum(ib, ith, global_inoutput, per_block_results);
    }
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

  // create array of 262144 elements
  const int num_elements = 1 << 18;
  uint num_blocks = num_elements / BLOCK_SIZE;

  // generate random input on the host
  std::vector<int> h_input(num_elements);
  for (auto& elt : h_input) {
    elt = distrib(gen);
  }

  // Compute the sum on the host
  const int host_result = std::accumulate(h_input.begin(), h_input.end(), 0);
  std::cerr << "Host sum: " << host_result << std::endl;

  // Input size
  uint in_size = num_elements * sizeof(int);


  // === CPU verification ===
  int host_partial_sums_and_total[num_blocks];
  if (num_blocks == BLOCK_SIZE) {
    std::cout << "OK, right num_blocks and BLOCK_SIZE\n";
  } else {
    std::cout << "ERROR - num_blocks(" << num_blocks << ") != BLOCK_SIZE(" << BLOCK_SIZE << ")\n";
    return 1;
  }
  cpu_reduction(h_input.data(), host_partial_sums_and_total, num_elements, num_blocks);
  int h_result;
  cpu_reduction(host_partial_sums_and_total, &h_result, BLOCK_SIZE, 1);
  std::cout << "Host reduce sum: " << h_result << std::endl;



  // === CUDA kernels ===

  // Move input to device memory
  int* d_input;
  cudaMalloc(&d_input, in_size);
  cudaMemcpy(d_input, h_input.data(), in_size, cudaMemcpyHostToDevice);

  // Allocate the partial sums: How much space does it need?
  int* d_partial_sums_and_total;
  cudaMalloc(&d_partial_sums_and_total, in_size);


  // Launch one kernel to compute, per-block, a partial sum. How
  // much shared memory does it need?
  block_sum<<<num_blocks, BLOCK_SIZE>>>(d_input, d_partial_sums_and_total,
                                        num_elements);

  // 1) Sommer les sommes partielles sur CPU
  // 2) Comparer les sommes partielles lues depuis le GPU et depuis le CPU

  int* d_result;
  cudaMalloc(&d_result, sizeof(int));

  // Compute the sum of the partial sums
  block_sum<<<1, BLOCKS_NUMBER>>>(d_partial_sums_and_total, d_result, BLOCKS_NUMBER);

  // Copy the result back to the host
  int device_result = 0;
  cudaMemcpy(&device_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);

  int h_partial_sums_and_total[num_blocks];
  cudaMemcpy(h_partial_sums_and_total, d_partial_sums_and_total, BLOCK_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
  int hsum = 0;
  for (uint ib = 0; ib < num_blocks; ++ib) {
    hsum += h_partial_sums_and_total[ib];
    //std::cout << "b(" << ib << ") = " << h_partial_sums_and_total[ib] << std::endl;
  }

  std::cout << "Device sum: " << device_result << std::endl;
  std::cout << "Host sum of partial sums: " << hsum << std::endl;


  std::cout << "\nPartial sums comparison: " << std::endl;
  for (uint ib = 0; ib < num_blocks; ++ib) {
    int gpu_v = h_partial_sums_and_total[ib];
    int cpu_v = host_partial_sums_and_total[ib];
    if (gpu_v == cpu_v) {
      std::cout << "-\n";
    } else {
      std::cout << "e[" << ib << ": " << gpu_v << "!=" << cpu_v << "]\n";
    }
    //std::cout << "b(" << ib << ") = " << h_partial_sums_and_total[ib] << std::endl;
  }
  


  // // Part 1 of 6: deallocate device memory

  return 0;
}
