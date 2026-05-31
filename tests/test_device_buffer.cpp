#include "curag/device_buffer.hpp"
#include "curag/cuda_utils.hpp"

#include <cassert>
#include <iostream>
#include <vector>

void test_resize_and_copy()
{
    curag::DeviceBuffer<float> buffer;

    assert(buffer.empty());
    assert(buffer.size() == 0);

    buffer.resize(4);

    assert(!buffer.empty());
    assert(buffer.size() == 4);

    std::vector<float> host = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> output(4, 0.0f);

    CURAG_CUDA_CHECK(cudaMemcpy(
        buffer.data(),
        host.data(),
        host.size() * sizeof(float),
        cudaMemcpyHostToDevice));

    CURAG_CUDA_CHECK(cudaMemcpy(
        output.data(),
        buffer.data(),
        output.size() * sizeof(float),
        cudaMemcpyDeviceToHost));

    for (std::size_t i = 0; i < host.size(); ++i)
    {
        assert(output[i] == host[i]);
    }
}

void test_resize_reuses_larger_buffer()
{
    curag::DeviceBuffer<float> buffer(10);

    float *old_ptr = buffer.data();

    buffer.resize(5);

    assert(buffer.size() == 10);
    assert(buffer.data() == old_ptr);
}

void test_resize_grows_buffer()
{
    curag::DeviceBuffer<float> buffer(4);

    buffer.resize(16);

    assert(buffer.size() == 16);
    assert(buffer.data() != nullptr);
}

void test_move_constructor()
{
    curag::DeviceBuffer<float> source(8);
    float *source_ptr = source.data();

    curag::DeviceBuffer<float> moved(std::move(source));

    assert(moved.size() == 8);
    assert(moved.data() == source_ptr);

    assert(source.size() == 0);
    assert(source.data() == nullptr);
}

int main()
{
    test_resize_and_copy();
    test_resize_reuses_larger_buffer();
    test_resize_grows_buffer();
    test_move_constructor();

    std::cout << "All DeviceBuffer tests passed.\n";
    return 0;
}