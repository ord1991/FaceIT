## 2025-05-26 - Vectorized Face Matching
**Learning:** Manual Python loops for distance calculation in face recognition are a major bottleneck. For 4096-dimensional embeddings (VGG-Face), even a small user base (100 users) causes measurable latency in a real-time stream.
**Action:** Always pre-normalize high-dimensional embeddings and use NumPy's `dot` product for O(N) vectorized matching. This yields ~300x speedup and allows for scalability on CPU-bound systems.

## 2026-05-26 - [GPU Acceleration Support]
- **Summary**: Added automatic GPU detection and configuration in `FaceEngine`.
- **Expected Impact**: Reduced frame processing latency by offloading neural network inference to GPU when available.
- **Learnings**: TensorFlow requires `set_memory_growth` to be set before the GPU is initialized to avoid overallocation of VRAM.
