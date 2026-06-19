#pragma once

namespace curag
{

    void batched_topk(
        const float *d_scores,
        float *d_topk_values,
        int *d_topk_indices,
        int num_queries,
        int num_vectors,
        int k);

}