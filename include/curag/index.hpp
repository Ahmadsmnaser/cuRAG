#pragma once

#include <vector>
#include <cstddef>

namespace curag
{

    struct SearchResult
    {
        std::vector<float> values;
        std::vector<int> indices;
    };

    class Index
    {
    public:
        explicit Index(int dim); // Constructor that initializes the index with the specified dimensionality
        ~Index(); // Destructor to clean up any allocated resources

        Index(const Index &) = delete; // Delete copy constructor to prevent copying of the index
        Index &operator=(const Index &) = delete; // Delete copy assignment operator to prevent copying of the index

        Index(Index &&other) noexcept; // Move constructor to allow moving of the index without copying
        Index &operator=(Index &&other) noexcept; // Move assignment operator to allow moving of the index without copying

        void build(const float *corpus, int num_vectors); // Method to build the index from a given corpus of vectors, where 'corpus' is a pointer to the vector data and 'num_vectors' is the number of vectors in the corpus

        SearchResult search(const float *query, int k) const; // Method to perform a search on the index using a given query vector, where 'query' is a pointer to the query vector and 'k' is the number of nearest neighbors to return

        int dim() const;
        int num_vectors() const;
        bool is_built() const;

    private:
        int dim_;
        int num_vectors_;
        float *d_corpus_;
    };

} // namespace curag