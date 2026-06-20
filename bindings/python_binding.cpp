#include "curag/index.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>

namespace py = pybind11;

namespace
{

    py::dict search_result_to_dict(const curag::SearchResult &result)
    {
        py::dict output;
        output["values"] = result.values;
        output["indices"] = result.indices;
        return output;
    }

    py::dict batch_search_result_to_dict(const curag::BatchSearchResult &result)
    {
        py::dict output;
        output["values"] = result.values;
        output["indices"] = result.indices;
        output["num_queries"] = result.num_queries;
        output["k"] = result.k;
        return output;
    }

} // namespace

PYBIND11_MODULE(curag, m)
{
    m.doc() = "cuRAG CUDA vector search bindings";

    py::class_<curag::Index>(m, "Index")
        .def(py::init<int>(), py::arg("dim"))

        .def("build", [](curag::Index &self, py::array_t<float, py::array::c_style | py::array::forcecast> corpus)
             {
                py::buffer_info info = corpus.request();

                if (info.ndim != 2) {
                    throw std::runtime_error("corpus must be a 2D float32 array");
                }

                int num_vectors = static_cast<int>(info.shape[0]);
                int dim = static_cast<int>(info.shape[1]);

                if (dim != self.dim()) {
                    throw std::runtime_error("corpus dimension does not match Index dimension");
                }

                const float* data = static_cast<const float*>(info.ptr);

                self.build(data, num_vectors); }, py::arg("corpus"))

        .def("search", [](const curag::Index &self, py::array_t<float, py::array::c_style | py::array::forcecast> query, int k)
             {
                py::buffer_info info = query.request();

                if (info.ndim != 1) {
                    throw std::runtime_error("query must be a 1D float32 array");
                }

                int dim = static_cast<int>(info.shape[0]);

                if (dim != self.dim()) {
                    throw std::runtime_error("query dimension does not match Index dimension");
                }

                const float* data = static_cast<const float*>(info.ptr);

                curag::SearchResult result = self.search(data, k);
                return search_result_to_dict(result); }, py::arg("query"), py::arg("k"))

        .def("search_batch", [](const curag::Index &self, py::array_t<float, py::array::c_style | py::array::forcecast> queries, int k)
             {
                py::buffer_info info = queries.request();

                if (info.ndim != 2) {
                    throw std::runtime_error("queries must be a 2D float32 array");
                }

                int num_queries = static_cast<int>(info.shape[0]);
                int dim = static_cast<int>(info.shape[1]);

                if (dim != self.dim()) {
                    throw std::runtime_error("queries dimension does not match Index dimension");
                }

                const float* data = static_cast<const float*>(info.ptr);

                curag::BatchSearchResult result =
                    self.search_batch(data, num_queries, k);

                return batch_search_result_to_dict(result); }, py::arg("queries"), py::arg("k"))

        .def("save", &curag::Index::save, py::arg("path"))

        .def_static("load", &curag::Index::load, py::arg("path"))

        .def_property_readonly("dim", &curag::Index::dim)
        .def_property_readonly("num_vectors", &curag::Index::num_vectors)
        .def_property_readonly("is_built", &curag::Index::is_built);
}