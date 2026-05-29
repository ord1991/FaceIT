## 2025-05-29 - Vectorized Matching and Async/Sync Optimization
**Learning:** Manual loops for cosine distance in Python are extremely slow. Pre-normalizing vectors and using `np.dot` provides a massive speedup (approx. 300x for matching). Also, for blocking I/O (like SQLAlchemy), using standard `def` in FastAPI is crucial to avoid blocking the event loop.
**Action:** Always pre-normalize embeddings and use vectorized NumPy operations for matching. Prefer standard `def` for endpoints performing blocking I/O.
