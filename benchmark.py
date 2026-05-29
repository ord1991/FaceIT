import time
import numpy as np
import cv2
from face_engine import FaceEngine, HAS_GPU
from database import init_db

def benchmark():
    init_db()
    engine = FaceEngine()
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(frame, "Benchmark", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Warmup
    engine.process_frame(frame)

    iterations = 5
    start_time = time.time()
    for _ in range(iterations):
        engine.process_frame(frame)
    end_time = time.time()

    avg_time = (end_time - start_time) / iterations
    device = "GPU" if HAS_GPU else "CPU"
    print(f"Average processing time per frame ({device}): {avg_time:.4f} seconds")

if __name__ == "__main__":
    benchmark()
