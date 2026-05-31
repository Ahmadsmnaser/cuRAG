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

void test_build_and_search()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 5;
    constexpr int k = 2;

    std::vector<float> corpus = {
        1.0f, 0.0f,  // index 0, score 1.0
        0.0f, 1.0f,  // index 1, score 0.0
        -1.0f, 0.0f, // index 2, score -1.0
        0.8f, 0.6f,  // index 3, score 0.8
        0.6f, 0.8f   // index 4, score 0.6
    };

    std::vector<float> query = {
        1.0f, 0.0f};

    curag::Index index(dim);
    index.build(corpus.data(), num_vectors);

    assert(index.is_built());
    assert(index.dim() == dim);
    assert(index.num_vectors() == num_vectors);

    curag::SearchResult result = index.search(query.data(), k);

    assert(result.values.size() == k);
    assert(result.indices.size() == k);

    expect_near(result.values[0], 1.0f);
    assert(result.indices[0] == 0);

    expect_near(result.values[1], 0.8f);
    assert(result.indices[1] == 3);
}

void test_search_before_build_fails()
{
    curag::Index index(2);

    std::vector<float> query = {
        1.0f, 0.0f};

    bool caught = false;

    try
    {
        index.search(query.data(), 1);
    }
    catch (const std::runtime_error &)
    {
        caught = true;
    }

    assert(caught);
}

void test_invalid_constructor()
{
    bool caught_zero_dim = false;

    try
    {
        curag::Index index(0);
    }
    catch (const std::invalid_argument &)
    {
        caught_zero_dim = true;
    }

    assert(caught_zero_dim);
}

void test_rebuild_index()
{
    constexpr int dim = 2;

    std::vector<float> corpus_a = {
        1.0f, 0.0f,
        0.0f, 1.0f};

    std::vector<float> corpus_b = {
        -1.0f, 0.0f,
        1.0f, 0.0f};

    std::vector<float> query = {
        1.0f, 0.0f};

    curag::Index index(dim);

    index.build(corpus_a.data(), 2);
    auto result_a = index.search(query.data(), 1);

    assert(result_a.indices[0] == 0);
    expect_near(result_a.values[0], 1.0f);

    index.build(corpus_b.data(), 2);
    auto result_b = index.search(query.data(), 1);

    assert(result_b.indices[0] == 1);
    expect_near(result_b.values[0], 1.0f);
}

int main()
{
    test_build_and_search();
    test_search_before_build_fails();
    test_invalid_constructor();
    test_rebuild_index();

    std::cout << "All Index tests passed.\n";
    return 0;
}
