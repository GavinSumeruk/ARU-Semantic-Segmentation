import time
from ultralytics import YOLO

def benchmark(model, image, runs=50, warmup=5):
    """Time a model over many runs. Discard warm-up runs, then average."""
    for _ in range(warmup):                 # warm-up runs (always slower; ignore)
        model(image, verbose=False)

    times = []
    for _ in range(runs):                   # timed runs
        start = time.time()
        model(image, verbose=False)
        times.append(time.time() - start)

    avg_s = sum(times) / len(times)
    print(f"average latency: {avg_s*1000:.1f} ms")
    print(f"throughput:      {1.0/avg_s:.1f} FPS")


if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    benchmark(model, "https://ultralytics.com/images/bus.jpg")