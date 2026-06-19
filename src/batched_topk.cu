#include "curag/batched_topk.hpp"
#include "curag/topk.hpp"

#include <stdexcept>
#include <cstddef>

namespace curag
{

    void batched_topk(
        const float *d_scores,
        float *d_topk_values,
        int *d_topk_indices,
        int num_queries,
        int num_vectors,
        int k)
    {
        if (d_scores == nullptr || d_topk_values == nullptr || d_topk_indices == nullptr)
        {
            throw std::runtime_error("batched_topk device pointers must not be null");
        }

        if (num_queries <= 0)
        {
            throw std::runtime_error("num_queries must be positive");
        }

        if (num_vectors <= 0)
        {
            throw std::runtime_error("num_vectors must be positive");
        }

        if (k <= 0)
        {
            throw std::runtime_error("k must be positive");
        }

        if (k > num_vectors)
        {
            throw std::runtime_error("k must be <= num_vectors");
        }

        for (int q = 0; q < num_queries; ++q)
        {
            const float *scores_for_query =
                d_scores + static_cast<std::size_t>(q) * num_vectors;

            float *values_for_query =
                d_topk_values + static_cast<std::size_t>(q) * k;

            int *indices_for_query =
                d_topk_indices + static_cast<std::size_t>(q) * k;

            topk(
                scores_for_query,
                values_for_query,
                indices_for_query,
                num_vectors,
                k);
        }
    }

} // namespace curag