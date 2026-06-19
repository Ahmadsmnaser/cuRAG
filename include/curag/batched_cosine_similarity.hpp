#pragma once

namespace curag
{

    void batched_cosine_similarity(
        const float *d_queries,
        const float *d_corpus,
        float *d_scores,
        int num_queries,
        int num_vectors,
        int dim);

}