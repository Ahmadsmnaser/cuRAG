import numpy as np
import curag

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

print("values:", result["values"])
print("indices:", result["indices"])

queries = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
    ],
    dtype=np.float32,
)

batch = index.search_batch(queries, 2)

print("batch values:", batch["values"])
print("batch indices:", batch["indices"])