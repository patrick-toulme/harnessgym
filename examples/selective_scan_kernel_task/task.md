# Custom Kernel Development Task

Implement the CPU reference kernel in `kernel.py`.

The function `selective_scan_kernel(u, delta, A, B, C, D)` is meant to behave like a custom fused selective-scan kernel: it updates a per-batch, per-channel recurrent state across the sequence dimension and returns the fused output tensor.

Requirements:
- Preserve the public function name and signature.
- Use NumPy only.
- Match the verifier's randomized numerical reference.
- Meet the verifier's performance threshold on the larger benchmark case.
- Run `python3 verifier.py` from this workspace and only report the task solved if it passes.
