import os
import tempfile

import numpy as np
import pytest
import curag


def test_build_and_search():
    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    query = np.array([1.0, 0.0], dtype=np.float32)

    index = curag.Index(2)
    index.build(corpus)

    result = index.search(query, 2)

    assert result["indices"] == [0, 3]
    assert np.allclose(result["values"], [1.0, 0.8], rtol=1e-5, atol=1e-5)


def test_search_batch():
    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    queries = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index = curag.Index(2)
    index.build(corpus)

    result = index.search_batch(queries, 2)

    assert result["num_queries"] == 2
    assert result["k"] == 2

    assert result["indices"] == [0, 3, 1, 4]
    assert np.allclose(
        result["values"],
        [1.0, 0.8, 1.0, 0.8],
        rtol=1e-5,
        atol=1e-5,
    )


def test_save_and_load():
    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [-1.0, 0.0],
            [0.8, 0.6],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    query = np.array([1.0, 0.0], dtype=np.float32)

    index = curag.Index(2)
    index.build(corpus)

    before = index.search(query, 2)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "index.curag")
        index.save(path)

        loaded = curag.Index.load(path)
        after = loaded.search(query, 2)

    assert before["indices"] == after["indices"]
    assert np.allclose(before["values"], after["values"], rtol=1e-5, atol=1e-5)


def test_invalid_shapes():
    index = curag.Index(2)

    bad_corpus = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(RuntimeError):
        index.build(bad_corpus)

    corpus = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    index.build(corpus)

    bad_query = np.array([[1.0, 0.0]], dtype=np.float32)

    with pytest.raises(RuntimeError):
        index.search(bad_query, 1)


def test_wrong_dimension():
    index = curag.Index(2)

    corpus = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(RuntimeError):
        index.build(corpus)