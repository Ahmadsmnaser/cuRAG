#pragma once

#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

namespace curag
{

    template <typename T>
    class DeviceBuffer
    {
    public:
        DeviceBuffer()
            : ptr_(nullptr),
              count_(0)
        {
        }

        explicit DeviceBuffer(std::size_t count)
            : ptr_(nullptr),
              count_(0)
        {
            resize(count);
        }

        ~DeviceBuffer()
        {
            release();
        }

        DeviceBuffer(const DeviceBuffer &) = delete;
        DeviceBuffer &operator=(const DeviceBuffer &) = delete;

        DeviceBuffer(DeviceBuffer &&other) noexcept
            : ptr_(other.ptr_),
              count_(other.count_)
        {
            other.ptr_ = nullptr;
            other.count_ = 0;
        }

        DeviceBuffer &operator=(DeviceBuffer &&other) noexcept
        {
            if (this != &other)
            {
                release();

                ptr_ = other.ptr_;
                count_ = other.count_;

                other.ptr_ = nullptr;
                other.count_ = 0;
            }

            return *this;
        }

        void resize(std::size_t new_count)
        {
            if (new_count <= count_)
            {
                return;
            }

            release();

            count_ = new_count;

            if (count_ > 0)
            {
                CURAG_CUDA_CHECK(cudaMalloc(&ptr_, count_ * sizeof(T)));
            }
        }

        void release()
        {
            if (ptr_ != nullptr)
            {
                cudaFree(ptr_);
                ptr_ = nullptr;
                count_ = 0;
            }
        }

        T *data()
        {
            return ptr_;
        }

        const T *data() const
        {
            return ptr_;
        }

        std::size_t size() const
        {
            return count_;
        }

        bool empty() const
        {
            return ptr_ == nullptr || count_ == 0;
        }

    private:
        T *ptr_;
        std::size_t count_;
    };

} // namespace curag