import time

import torch


DIM = 768
WARMUP_RUNS = 5
MEASURED_RUNS = 20


def run_benchmark(num_vectors: int) -> None:
    query = torch.rand(DIM, device="cuda", dtype=torch.float32) * 2.0 - 1.0
    corpus = (
        torch.rand(num_vectors, DIM, device="cuda", dtype=torch.float32) * 2.0
        - 1.0
    )

    query = torch.nn.functional.normalize(query, dim=0)
    corpus = torch.nn.functional.normalize(corpus, dim=1)

    for _ in range(WARMUP_RUNS):
        torch.mv(corpus, query)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(MEASURED_RUNS):
        torch.mv(corpus, query)
    stop.record()
    torch.cuda.synchronize()

    average_ms = start.elapsed_time(stop) / MEASURED_RUNS
    vectors_per_second = num_vectors / (average_ms / 1000.0)

    print(
        f"{num_vectors:>10}{DIM:>10}{average_ms:>16.3f}"
        f"{vectors_per_second:>20.0f}"
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA support is required")

    print("PyTorch torch.mv CUDA baseline")
    print(f"{'vectors':>10}{'dim':>10}{'kernel_ms':>16}{'vectors_per_sec':>20}")

    run_benchmark(10_000)
    run_benchmark(100_000)


if __name__ == "__main__":
    start = time.perf_counter()
    main()
    print(f"Total script time: {time.perf_counter() - start:.2f}s")
