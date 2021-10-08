#include <random>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>
#include <numeric>
#include <execution>
#include <chrono>

int main()
{
  // define a pseudo-random number generator engine and seed it using an actual
  // random device
  std::random_device rd;
  std::default_random_engine eng{rd()};

  int const MAX_N = 100;
  std::uniform_int_distribution<int> uniform_dist{1, MAX_N};

  // fill a vector with SIZE random numbers
  int const SIZE = 10000000;
  std::vector<int> v;
  v.reserve(SIZE);
  std::generate_n(std::back_inserter(v), SIZE, [&] { return uniform_dist(eng); });

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::accumulate
    int sum = std::accumulate(v.begin(), v.end(), 0, [](int &a, int &b) { return a + b; });
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << " in " << d.count() << " s  sum(" + std::to_string(sum) + ")\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, sequential policy
    // NB you need to pass the initial value
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sum all the elements of the vector with std::reduce, parallel policy
    // NB you need to pass the initial value
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << " in " << d.count() << " s\n";
  }

  {
    auto t0 = std::chrono::high_resolution_clock::now();
    // sort the vector with std::sort, parallel policy
    auto t1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> d = t1 - t0;
    std::cout << " in " << d.count() << " s\n";
  }
}
