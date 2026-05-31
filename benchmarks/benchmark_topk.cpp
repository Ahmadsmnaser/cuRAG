namespace {

    void topk(
        const float* d_scores,
        float* d_topk_values,
        int* d_topk_indices,
        int num_scores,
        int k
    );

}