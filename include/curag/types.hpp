#pragma once

#include <cstddef>

namespace curag{

    using index_t = int;
    using size_t = std::size_t;

    struct SearchConfig{
        int dims;
        int num_vectors;
    };
}    