#include <iostream>
#include <iomanip>

#include <random>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <chrono>

#include <cmath>
//#include <ctgmath>

double compute_part(uint thread_id, uint num_steps, uint iter_per_thread, uint threads_nb) {

  start_i = thread_id * iter_per_thread;
  if (thread_id + 1 != threads_nb) 
  uint start_i, uint stop_i;


  double lsum = 0;
  for (int i = start_i; i < stop_i; ++i){
    double x = (i + 0.5) / num_steps;
    lsum = lsum + 4.0 / (1.0 + x*x);
  }
  return lsum;
}


int main()
{
  double sum = 0.;
  constexpr unsigned int num_steps = 1 << 22;
  double pi = 0.0;
  constexpr double step = 1.0/(double) num_steps;
  auto start = std::chrono::system_clock::now();

  uint thread_nb = oneapi::tbb::this_task_arena::max_concurrency();

  double part_sums[thread_nb];
  double iter_per_thread = std::ceil(double(num_steps) / double(thread_nb));

  for (int i=0; i < num_steps; i++){
    auto  x = (i+0.5)/num_steps;
    sum = sum + 4.0/(1.0+x*x);
  }

  std::vector<std::thread> threads;
  threads.reserve(thread_nb);
  for (uint it = 0; it < thread_nb; ++it) {
    threads.emplace_back();
  }
  //Construct a thread which runs the function f
  std::thread t0(f,0);

  //and then destroy it by joining it
  t0.join();


  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> dur= stop - start;
  std::cout << dur.count() << " seconds" << std::endl;
  pi = step * sum;

  std::cout << "result: " <<  std::setprecision (15) << pi << std::endl;
}
