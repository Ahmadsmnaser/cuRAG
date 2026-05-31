#include "curag/index.hpp"

#include "curag/cosine_similarity.hpp"
#include "curag/normalize.hpp"
#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <utility>

namespace curag
{

    Index::Index(int dim)
        : dim_(dim), num_vectors_(0), d_corpus_(nullptr)
    {
        if (dim <= 0)
        {
            throw std::invalid_argument("Dimension must be positive");
        }
        if (dim > 1024)
        {
            throw std::invalid_argument("Dimension exceeds maximum supported value (1024)");
        }
    }
    Index::~Index()
    {
        if (d_corpus_)
        {
            cudaFree(d_corpus_);
            d_corpus_ = nullptr;
        }
    }

    Index::Index(Index &&other) noexcept
        : dim_(other.dim_), num_vectors_(other.num_vectors_), d_corpus_(other.d_corpus_)
    {
        other.dim_ = 0;
        other.num_vectors_ = 0;
        other.d_corpus_ = nullptr;
    }

    Index &Index::operator=(Index &&other) noexcept
    {
        if (this != &other)
        {
            if (d_corpus_)
            {
                cudaFree(d_corpus_);
            }
            dim_ = other.dim_;
            num_vectors_ = other.num_vectors_;
            d_corpus_ = other.d_corpus_;

            other.dim_ = 0;
            other.num_vectors_ = 0;
            other.d_corpus_ = nullptr;
        }
        return *this;
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
        if (d_corpus_ != nullptr)
        {
            CURAG_CUDA_CHECK(cudaFree(d_corpus_));
            d_corpus_ = nullptr;
        }
        num_vectors_ = num_vectors;
        size_t corpus_bytes = static_cast<size_t>(dim_) * num_vectors_ * sizeof(float);
        CURAG_CUDA_CHECK(cudaMalloc(&d_corpus_, corpus_bytes));

        CURAG_CUDA_CHECK(cudaMemcpy(d_corpus_, corpus, corpus_bytes, cudaMemcpyHostToDevice));

        // Normalize the corpus vectors in-place on the GPU - ONCE
        l2_normalize(d_corpus_, num_vectors_, dim_);

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

        float *d_query = nullptr;
        float *d_scores = nullptr;
        float *d_topk_values = nullptr;
        int *d_topk_indices = nullptr;

        size_t query_bytes = dim_ * sizeof(float);
        size_t scores_bytes = num_vectors_ * sizeof(float);
        size_t topk_values_bytes = k * sizeof(float);
        size_t topk_indices_bytes = k * sizeof(int);

        CURAG_CUDA_CHECK(cudaMalloc(&d_query, query_bytes));
        CURAG_CUDA_CHECK(cudaMalloc(&d_scores, scores_bytes));
        CURAG_CUDA_CHECK(cudaMalloc(&d_topk_values, topk_values_bytes));
        CURAG_CUDA_CHECK(cudaMalloc(&d_topk_indices, topk_indices_bytes));

        CURAG_CUDA_CHECK(cudaMemcpy(
            d_query,
            query,
            query_bytes,
            cudaMemcpyHostToDevice));

        l2_normalize(d_query, 1, dim_);

        cosine_similarity(
            d_query,
            d_corpus_,
            d_scores,
            num_vectors_,
            dim_);

        topk(
            d_scores,
            d_topk_values,
            d_topk_indices,
            num_vectors_,
            k);

        CURAG_CUDA_CHECK(cudaDeviceSynchronize());

        SearchResult result;
        result.values.resize(k);
        result.indices.resize(k);

        CURAG_CUDA_CHECK(cudaMemcpy(
            result.values.data(),
            d_topk_values,
            topk_values_bytes,
            cudaMemcpyDeviceToHost));

        CURAG_CUDA_CHECK(cudaMemcpy(
            result.indices.data(),
            d_topk_indices,
            topk_indices_bytes,
            cudaMemcpyDeviceToHost));

        CURAG_CUDA_CHECK(cudaFree(d_query));
        CURAG_CUDA_CHECK(cudaFree(d_scores));
        CURAG_CUDA_CHECK(cudaFree(d_topk_values));
        CURAG_CUDA_CHECK(cudaFree(d_topk_indices));

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
        return d_corpus_ != nullptr && num_vectors_ > 0;
    }
}