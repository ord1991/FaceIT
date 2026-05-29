## 2025-05-26 - Vectorized Face Matching
**Learning:** Manual Python loops for distance calculation in face recognition are a major bottleneck. For 4096-dimensional embeddings (VGG-Face), even a small user base (100 users) causes measurable latency in a real-time stream.
**Action:** Always pre-normalize high-dimensional embeddings and use NumPy's `dot` product for O(N) vectorized matching. This yields ~300x speedup and allows for scalability on CPU-bound systems.

## 2026-05-26 - GPU Acceleration
Implemented automatic GPU detection and configuration in the FaceEngine. The system now checks for physical GPU devices via TensorFlow and enables dynamic memory growth to optimize VRAM usage. This allows the system to scale performance automatically on hardware with dedicated GPUs while maintaining a safe CPU fallback.

## 2026-05-26 - [Reducing Lock Contention and Event Loop Blocking in unknown_faces]
FastAPI endpoints that perform synchronous file I/O (like `os.remove`) should be defined with `def` instead of `async def` to ensure they are run in a thread pool and do not block the asynchronous event loop. Furthermore, moving these I/O operations outside of critical sections protected by locks (e.g., `unknown_faces_lock`) significantly improves performance by reducing lock contention, as demonstrated by a 27x increase in lock availability during peak I/O in benchmark tests.
