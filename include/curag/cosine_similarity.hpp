#pragma once

namespace curag
{

    void cosine_similarity(
        const float *d_query,
        const float *d_corpus,
        float *d_scores,
        int num_vectors,
        int dim);

}