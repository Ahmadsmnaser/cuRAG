#include "curag/index.hpp"
#include "curag/cosine_similarity.hpp"
#include "curag/normalize.hpp"
#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace curag
{

    Index::Index(int dim)
        : dim_(dim),
          num_vectors_(0)
    {
        if (dim_ <= 0)
        {
            throw std::invalid_argument("Index dim must be positive");
        }

        if (dim_ > 1024)
        {
            throw std::invalid_argument("Index dim exceeds maximum supported dimension 1024");
        }
    }

    void Index::build(const float *corpus, int num_vectors)
    {
        if (corpus == nullptr)
        {
            throw std::runtime_error("corpus must not be null");
        }

        if (num_vectors <= 0)
        {
            throw std::runtime_error("num_vectors must be positive");
        }

        num_vectors_ = num_vectors;

        std::size_t corpus_count =
            static_cast<std::size_t>(num_vectors_) * dim_;

        corpus_buffer_.resize(corpus_count);

        CURAG_CUDA_CHECK(cudaMemcpy(
            corpus_buffer_.data(),
            corpus,
            corpus_count * sizeof(float),
            cudaMemcpyHostToDevice));

        // Normalize corpus once at index build time.
        l2_normalize(corpus_buffer_.data(), num_vectors_, dim_);

        CURAG_CUDA_CHECK(cudaDeviceSynchronize());
    }

    SearchResult Index::search(const float *query, int k) const
    {
        if (!is_built())
        {
            throw std::runtime_error("Index must be built before search");
        }

        if (query == nullptr)
        {
            throw std::runtime_error("query must not be null");
        }

        if (k <= 0)
        {
            throw std::runtime_error("k must be positive");
        }

        if (k > num_vectors_)
        {
            throw std::runtime_error("k must be <= num_vectors");
        }

        query_buffer_.resize(static_cast<std::size_t>(dim_));
        scores_buffer_.resize(static_cast<std::size_t>(num_vectors_));
        topk_values_buffer_.resize(static_cast<std::size_t>(k));
        topk_indices_buffer_.resize(static_cast<std::size_t>(k));

        CURAG_CUDA_CHECK(cudaMemcpy(
            query_buffer_.data(),
            query,
            static_cast<std::size_t>(dim_) * sizeof(float),
            cudaMemcpyHostToDevice));

        l2_normalize(query_buffer_.data(), 1, dim_);

        cosine_similarity(
            query_buffer_.data(),
            corpus_buffer_.data(),
            scores_buffer_.data(),
            num_vectors_,
            dim_);

        topk(
            scores_buffer_.data(),
            topk_values_buffer_.data(),
            topk_indices_buffer_.data(),
            num_vectors_,
            k);

        CURAG_CUDA_CHECK(cudaDeviceSynchronize());

        SearchResult result;
        result.values.resize(k);
        result.indices.resize(k);

        CURAG_CUDA_CHECK(cudaMemcpy(
            result.values.data(),
            topk_values_buffer_.data(),
            static_cast<std::size_t>(k) * sizeof(float),
            cudaMemcpyDeviceToHost));

        CURAG_CUDA_CHECK(cudaMemcpy(
            result.indices.data(),
            topk_indices_buffer_.data(),
            static_cast<std::size_t>(k) * sizeof(int),
            cudaMemcpyDeviceToHost));

        return result;
    }

    int Index::dim() const
    {
        return dim_;
    }

    int Index::num_vectors() const
    {
        return num_vectors_;
    }

    bool Index::is_built() const
    {
        return !corpus_buffer_.empty() && num_vectors_ > 0;
    }

} // namespace curag
