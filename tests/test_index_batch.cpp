#include "curag/index.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

void expect_near(float actual, float expected, float eps = 1e-5f)
{
    assert(std::fabs(actual - expected) <= eps);
}

void test_batch_search_matches_individual_search()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 5;
    constexpr int num_queries = 3;
    constexpr int k = 2;

    std::vector<float> corpus = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f,
        0.8f, 0.6f,
        0.6f, 0.8f};

    std::vector<float> queries = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f};

    curag::Index index(dim);
    index.build(corpus.data(), num_vectors);

    curag::BatchSearchResult batch =
        index.search_batch(queries.data(), num_queries, k);

    assert(batch.num_queries == num_queries);
    assert(batch.k == k);
    assert(batch.values.size() == static_cast<std::size_t>(num_queries) * k);
    assert(batch.indices.size() == static_cast<std::size_t>(num_queries) * k);

    for (int q = 0; q < num_queries; ++q)
    {
        const float *query =
            queries.data() + static_cast<std::size_t>(q) * dim;

        curag::SearchResult single = index.search(query, k);

        for (int j = 0; j < k; ++j)
        {
            std::size_t pos = static_cast<std::size_t>(q) * k + j;

            expect_near(batch.values[pos], single.values[j]);
            assert(batch.indices[pos] == single.indices[j]);
        }
    }
}

void test_batch_invalid_inputs()
{
    curag::Index index(2);

    std::vector<float> corpus = {
        1.0f, 0.0f,
        0.0f, 1.0f};

    std::vector<float> queries = {
        1.0f, 0.0f};

    bool caught_before_build = false;
    try
    {
        index.search_batch(queries.data(), 1, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_before_build = true;
    }
    assert(caught_before_build);

    index.build(corpus.data(), 2);

    bool caught_null_queries = false;
    try
    {
        index.search_batch(nullptr, 1, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_null_queries = true;
    }
    assert(caught_null_queries);

    bool caught_zero_queries = false;
    try
    {
        index.search_batch(queries.data(), 0, 1);
    }
    catch (const std::runtime_error &)
    {
        caught_zero_queries = true;
    }
    assert(caught_zero_queries);

    bool caught_bad_k = false;
    try
    {
        index.search_batch(queries.data(), 1, 0);
    }
    catch (const std::runtime_error &)
    {
        caught_bad_k = true;
    }
    assert(caught_bad_k);
}

int main()
{
    test_batch_search_matches_individual_search();
    test_batch_invalid_inputs();

    std::cout << "All Index batch search tests passed.\n";
    return 0;
}