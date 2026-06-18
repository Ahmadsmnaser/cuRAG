#include "curag/index.hpp"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <vector>

void expect_near(float actual, float expected, float eps = 1e-5f)
{
    assert(std::fabs(actual - expected) <= eps);
}

void test_save_and_load_index()
{
    constexpr int dim = 2;
    constexpr int num_vectors = 5;
    constexpr int k = 2;

    const std::string path = "tmp_test_index.curag";

    std::vector<float> corpus = {
        1.0f, 0.0f,
        0.0f, 1.0f,
        -1.0f, 0.0f,
        0.8f, 0.6f,
        0.6f, 0.8f};

    std::vector<float> query = {
        1.0f, 0.0f};

    curag::Index index(dim);
    index.build(corpus.data(), num_vectors);

    auto before = index.search(query.data(), k);

    index.save(path);

    curag::Index loaded = curag::Index::load(path);

    assert(loaded.is_built());
    assert(loaded.dim() == dim);
    assert(loaded.num_vectors() == num_vectors);

    auto after = loaded.search(query.data(), k);

    assert(before.values.size() == after.values.size());
    assert(before.indices.size() == after.indices.size());

    for (int i = 0; i < k; ++i)
    {
        expect_near(after.values[i], before.values[i]);
        assert(after.indices[i] == before.indices[i]);
    }

    std::remove(path.c_str());
}

void test_save_unbuilt_index_fails()
{
    curag::Index index(2);

    bool caught = false;

    try
    {
        index.save("should_not_exist.curag");
    }
    catch (const std::runtime_error &)
    {
        caught = true;
    }

    assert(caught);
}

void test_load_missing_file_fails()
{
    bool caught = false;

    try
    {
        auto index = curag::Index::load("missing_index_file.curag");
        (void)index;
    }
    catch (const std::runtime_error &)
    {
        caught = true;
    }

    assert(caught);
}

int main()
{
    test_save_and_load_index();
    test_save_unbuilt_index_fails();
    test_load_missing_file_fails();

    std::cout << "All Index save/load tests passed.\n";
    return 0;
}