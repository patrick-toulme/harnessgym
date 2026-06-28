# Hard Custom Kernel Development Task

Implement `paged_attention_kernel` in `kernel.py`.

This is a CPU reference version of a paged-attention decode kernel. The task intentionally mirrors the kind of indexing, masking, grouped-query head mapping, logit post-processing, and objective benchmarking that makes custom GPU kernel work slow to diagnose.

Requirements:
- Preserve the public function name and signature.
- Use NumPy only.
- Return both the output tensor and per-row log-sum-exp tensor exactly as the verifier expects.
- Derive the precise semantics from `verifier.py`; it is the objective source of truth.
- Pass randomized accuracy checks and the larger performance benchmark.
- Run `python3 verifier.py` from this workspace and only report solved when it passes.
