#include "curag/index.hpp"
#include "curag/cosine_similarity.hpp"
#include "curag/normalize.hpp"
#include "curag/topk.hpp"
#include "curag/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

#include <fstream>
#include <vector>
#include <cstdint>
#include <cstring>

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

    BatchSearchResult Index::search_batch(const float *queries, int num_queries, int k) const{
        if (!is_built())
        {
            throw std::runtime_error("Index must be built before batch search");
        }

        if (queries == nullptr)
        {
            throw std::runtime_error("queries must not be null");
        }

        if (num_queries <= 0)
        {
            throw std::runtime_error("num_queries must be positive");
        }

        if (k <= 0)
        {
            throw std::runtime_error("k must be positive");
        }

        if (k > num_vectors_)
        {
            throw std::runtime_error("k must be <= num_vectors");
        }

        BatchSearchResult batch;
        batch.num_queries = num_queries;
        batch.k = k;

        batch.values.resize(static_cast<std::size_t>(num_queries) * k);
        batch.indices.resize(static_cast<std::size_t>(num_queries) * k);

        for (int q = 0; q < num_queries; ++q)
        {
            const float *query =
                queries + static_cast<std::size_t>(q) * dim_;

            SearchResult result = search(query, k);

            for (int j = 0; j < k; ++j)
            {
                std::size_t out_index =
                    static_cast<std::size_t>(q) * k + j;

                batch.values[out_index] = result.values[j];
                batch.indices[out_index] = result.indices[j];
            }
        }

        return batch;
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

    namespace
    {
        constexpr char INDEX_MAGIC[8] = { 'C', 'U', 'R', 'A', 'G', 'I', 'D', 'N'};
        constexpr std::uint32_t INDEX_VERSION = 1;
    }

    void Index::save(const std::string &path) const
    {
        if(!is_built())
        {
            throw std::runtime_error("Index must be built before saving");
        }
        std::ofstream ofs(path, std::ios::binary);
        if(!ofs)
        {
            throw std::runtime_error("Failed to open file for writing: " + path);
        }
        std::int32_t dim_i32 = static_cast<std::int32_t>(dim_);
        std::int32_t num_vectors_i32 = static_cast<std::int32_t>(num_vectors_);

        ofs.write(INDEX_MAGIC, sizeof(INDEX_MAGIC));
        ofs.write(reinterpret_cast<const char*>(&INDEX_VERSION), sizeof(INDEX_VERSION));
        ofs.write(reinterpret_cast<const char*>(&dim_i32), sizeof(dim_i32));
        ofs.write(reinterpret_cast<const char*>(&num_vectors_i32), sizeof(num_vectors_i32));

        std::size_t corpus_count =
            static_cast<std::size_t>(num_vectors_) * dim_;

        std::vector<float> host_corpus(corpus_count);

        CURAG_CUDA_CHECK(cudaMemcpy(
            host_corpus.data(),
            corpus_buffer_.data(),
            corpus_count * sizeof(float),
            cudaMemcpyDeviceToHost));

        ofs.write(reinterpret_cast<const char*>(host_corpus.data()), corpus_count * sizeof(float));
        if(!ofs)
        {
            throw std::runtime_error("Failed to write index data to file: " + path);
        }
    }

    Index Index::load(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in)
        {
            throw std::runtime_error("Failed to open index file for reading");
        }

        char magic[8];
        std::uint32_t version = 0;
        std::int32_t dim = 0;
        std::int32_t num_vectors = 0;

        in.read(magic, sizeof(magic));
        in.read(reinterpret_cast<char *>(&version), sizeof(version));
        in.read(reinterpret_cast<char *>(&dim), sizeof(dim));
        in.read(reinterpret_cast<char *>(&num_vectors), sizeof(num_vectors));

        if (!in)
        {
            throw std::runtime_error("Invalid or truncated index header");
        }

        if (std::memcmp(magic, INDEX_MAGIC, sizeof(INDEX_MAGIC)) != 0)
        {
            throw std::runtime_error("Invalid cuRAG index file magic");
        }

        if (version != INDEX_VERSION)
        {
            throw std::runtime_error("Unsupported cuRAG index version");
        }

        if (dim <= 0 || dim > 1024)
        {
            throw std::runtime_error("Invalid dimension in index file");
        }

        if (num_vectors <= 0)
        {
            throw std::runtime_error("Invalid vector count in index file");
        }

        std::size_t corpus_count =
            static_cast<std::size_t>(num_vectors) * dim;

        std::vector<float> host_corpus(corpus_count);

        in.read(
            reinterpret_cast<char *>(host_corpus.data()),
            static_cast<std::streamsize>(corpus_count * sizeof(float)));

        if (!in)
        {
            throw std::runtime_error("Invalid or truncated index corpus data");
        }

        Index index(dim);

        index.num_vectors_ = num_vectors;
        index.corpus_buffer_.resize(corpus_count);

        CURAG_CUDA_CHECK(cudaMemcpy(
            index.corpus_buffer_.data(),
            host_corpus.data(),
            corpus_count * sizeof(float),
            cudaMemcpyHostToDevice));

        CURAG_CUDA_CHECK(cudaDeviceSynchronize());

        return index;
    }


} // namespace curag
