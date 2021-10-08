#include <cstdint>
#include <oneapi/tbb.h>
#include <oneapi/tbb/info.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task_arena.h>
#include <cassert>
#include <iostream>

int main() {
  // Get the default number of threads
  int num_threads = oneapi::tbb::info::default_concurrency();

  // Run the default parallelism
  oneapi::tbb::parallel_for(
      oneapi::tbb::blocked_range<size_t>(0, 20),
      [=](const oneapi::tbb::blocked_range<size_t> &r) {
        // Assert the maximum number of threads
        assert(num_threads == oneapi::tbb::this_task_arena::max_concurrency());
      });

  // Create the default task_arena
  oneapi::tbb::task_arena arena;
  arena.execute([=] {
    oneapi::tbb::parallel_for(
        oneapi::tbb::blocked_range<size_t>(0, 20),
        [=](const oneapi::tbb::blocked_range<size_t> &r) {
          // Assert the maximum number of threads
          assert(num_threads ==
                 oneapi::tbb::this_task_arena::max_concurrency());
        });
  });

  std::cout << "max_concurrency = " << oneapi::tbb::this_task_arena::max_concurrency() << std::endl;

  return 0;
}