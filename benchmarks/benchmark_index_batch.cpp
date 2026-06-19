#include "curag/index.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{

constexpr int kWarmupQueries = 5;

struct BenchmarkResult
{
    double total_ms;
    double latency_ms;
    double queries_per_second;
};

void fill_random(std::vector<float> &values)
{
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (float &value : values)
    {
        value = dist(rng);
    }
}

BenchmarkResult measure_repeated_search(
    const curag::Index &index,
    const std::vector<float> &queries,
    int num_queries,
    int dim,
    int k)
{
    auto start = std::chrono::steady_clock::now();

    for (int q = 0; q < num_queries; ++q)
    {
        const float *query =
            queries.data() + static_cast<std::size_t>(q) * dim;
        index.search(query, k);
    }

    auto stop = std::chrono::steady_clock::now();
    double total_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    return {
        total_ms,
        total_ms / num_queries,
        num_queries * 1000.0 / total_ms};
}

BenchmarkResult measure_batch_search(
    const curag::Index &index,
    const std::vector<float> &queries,
    int num_queries,
    int k)
{
    auto start = std::chrono::steady_clock::now();
    index.search_batch(queries.data(), num_queries, k);
    auto stop = std::chrono::steady_clock::now();

    double total_ms =
        std::chrono::duration<double, std::milli>(stop - start).count();

    return {
        total_ms,
        total_ms / num_queries,
        num_queries * 1000.0 / total_ms};
}

void print_result(const char *label, const BenchmarkResult &result)
{
    std::cout << "  " << std::left << std::setw(18) << label
              << std::right << std::fixed << std::setprecision(3)
              << " total=" << std::setw(10) << result.total_ms << " ms"
              << "  latency=" << std::setw(8) << result.latency_ms
              << " ms/query"
              << "  throughput=" << std::setw(9)
              << result.queries_per_second << " queries/sec\n";
}

void run_benchmark(int num_vectors, int dim, int k, int num_queries)
{
    std::vector<float> corpus(
        static_cast<std::size_t>(num_vectors) * dim);
    std::vector<float> queries(
        static_cast<std::size_t>(num_queries) * dim);

    fill_random(corpus);
    fill_random(queries);

    curag::Index index(dim);
    index.build(corpus.data(), num_vectors);

    for (int q = 0; q < kWarmupQueries; ++q)
    {
        index.search(
            queries.data() + static_cast<std::size_t>(q) * dim,
            k);
    }
    index.search_batch(queries.data(), kWarmupQueries, k);

    BenchmarkResult repeated = measure_repeated_search(
        index, queries, num_queries, dim, k);
    BenchmarkResult batch = measure_batch_search(
        index, queries, num_queries, k);

    std::cout << "N=" << num_vectors
              << " dim=" << dim
              << " K=" << k
              << " queries=" << num_queries << '\n';

    print_result("repeated search", repeated);
    print_result("search_batch", batch);

    std::cout << "  batch speedup:     "
              << std::fixed << std::setprecision(3)
              << repeated.total_ms / batch.total_ms << "x\n\n";
}

} // namespace

int main()
{
    constexpr int dim = 768;
    constexpr int k = 10;
    constexpr int num_queries = 100;

    run_benchmark(10'000, dim, k, num_queries);
    run_benchmark(100'000, dim, k, num_queries);
    return 0;
}
