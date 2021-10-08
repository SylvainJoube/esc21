#include <iostream>
#include <iomanip>

#include <random>
#include <vector>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <chrono>

// OneAPI
#include <oneapi/tbb.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>

//#include <>


int main()
{
  double sum = 0.;
  constexpr unsigned int num_steps = 1 << 22;
  double pi = 0.0;
  constexpr double step = 1.0/(double) num_steps;
  auto start = std::chrono::system_clock::now();

  for (int i=0; i < num_steps; i++){
    auto  x = (i+0.5)/num_steps;
    sum = sum + 4.0/(1.0+x*x);
  }

  const int N = oneapi::tbb::this_task_arena::max_concurrency();

  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<int>(0,N,<G>),
      [&](const oneapi::tbb::blocked_range<int>& range) {

        for(int i = range.begin(); i< range.end(); ++i) {
          //x[i]++;
        }

      }//,
      //<partitioner>
    );



  auto stop = std::chrono::system_clock::now();
  std::chrono::duration<double> dur= stop - start;
  std::cout << dur.count() << " seconds" << std::endl;
  pi = step * sum;

  std::cout << "result: " <<  std::setprecision (15) << pi << std::endl;
}
