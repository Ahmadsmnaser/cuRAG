#include "curag/index.hpp"

#include <chrono>
#include <iostream>
#include <random>
#include <vector>

void fill_random(std::vector<float> &values)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &value : values)
    {
        value = dist(rng);
    }
}

void run_benchmark(int num_vectors, int dim, int k, int num_queries)
{
    std::vector<float> corpus(static_cast<std::size_t>(num_vectors) * dim);
    std::vector<float> queries(static_cast<std::size_t>(num_queries) * dim);

    fill_random(corpus);
    fill_random(queries);

    curag::Index index(dim);
    index.build(corpus.data(), num_vectors);

    // Warmup
    for (int i = 0; i < 5; ++i)
    {
        index.search(queries.data() + static_cast<std::size_t>(i) * dim, k);
    }

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_queries; ++i)
    {
        index.search(queries.data() + static_cast<std::size_t>(i) * dim, k);
    }

    auto stop = std::chrono::high_resolution_clock::now();

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    double avg_latency_ms = elapsed_ms / num_queries;
    double qps = 1000.0 / avg_latency_ms;

    std::cout << "N=" << num_vectors
              << " dim=" << dim
              << " K=" << k
              << " queries=" << num_queries << "\n";

    std::cout << "  avg_latency: " << avg_latency_ms << " ms/query\n";
    std::cout << "  throughput:  " << qps << " queries/sec\n\n";
}

int main()
{
    constexpr int dim = 768;
    constexpr int k = 10;
    constexpr int num_queries = 100;

    run_benchmark(10'000, dim, k, num_queries);
    run_benchmark(100'000, dim, k, num_queries);

    return 0;
}