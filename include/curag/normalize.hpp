#pragma once

namespace curag
{

    void l2_normalize(
        float *d_vectors,
        int num_vectors,
        int dim);

}