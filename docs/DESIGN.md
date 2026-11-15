# Design and parameter accounting

`GPT.forward(idx, targets=None)` maps integer IDs `(B,T)` to logits `(B,T,V)`
and optional scalar cross entropy. Training rejects empty or oversized context;
generation crops to the last `block_size` tokens and restarts learned position IDs.

```text
IDs (B,T) -> token embedding + optional position embedding -> dropout
  -> L times:
     x = x + Dropout(Attention(LayerNorm(x)))
     x = x + Dropout(Linear(GELU(Linear(LayerNorm(x)))))
  -> final LayerNorm -> tied output projection -> logits (B,T,V)
```

Attention projects to `(B,T,3E)`, splits Q/K/V, reshapes each to `(B,H,T,E/H)`,
computes `Q @ K.transpose(-2,-1) / sqrt(E/H)`, fills future scores with `-inf`,
then applies softmax, attention dropout, and multiplication by V. Heads are
concatenated before the residual output projection. The optional SDPA path is
checked against the manual path for both outputs and gradients with dropout off.
There is no library Transformer or MultiheadAttention implementation.

## Exact trainable parameter count

Let E be embedding width, R the MLP expansion ratio, L the number of blocks,
V vocabulary size, T context size, b the bias switch (0/1), and p the learned
position switch (0/1).

- Tied input/output embedding: `V*E` (count once).
- Learned positions: `p*T*E`.
- Attention matrices: `3E² + E² = 4E²` per block.
- MLP matrices: `R*E² + R*E² = 2R*E²` per block.
- Two LayerNorm scale vectors: `2E` per block.
- Attention, MLP, and two LayerNorm biases: `b*(R+7)*E` per block.
- Final LayerNorm: `(1+b)*E`.

```text
N = V*E + p*T*E + L*((4+2R)*E² + 2E + b*(R+7)*E) + (1+b)*E
```

With R=4 and b=p=1 this becomes `V*E + T*E + L*(12E² + 13E) + 2E`.
For V=65: Tiny has **108,352** parameters; Small has **10,770,816**.
`tests/test_model.py` verifies this formula against actual unique parameters,
including disabling positions and biases. All weights use N(0, 0.02); biases
start at zero; attention and MLP residual output matrices use standard deviation
`0.02/sqrt(2L)`. LayerNorm scales retain the standard initial value one.

## Reproducibility boundaries

Checkpoint files include resolved configuration, dataset SHA-256, Git hash,
optimizer moments, global Python/NumPy/PyTorch RNGs, batch generator state,
active accelerator RNG, completed steps, best validation loss, and history.
The LR schedule is a pure function of completed steps and the original config.
Use `--stop-after` to interrupt without changing the LR horizon.

Evaluation batches use a freshly seeded independent generator; evaluation mode
turns off dropout and restores the previous model mode afterward. The resume
test is bit-for-bit on CPU, including dropout. Cross-device and cross-version
bitwise equality is not promised. FP32 is the default on MPS/CPU; optional BF16
requires CUDA hardware that reports BF16 support. There is no torch.compile.

Training throughput excludes evaluation/checkpoint work between intervals;
reported training elapsed time includes earlier checkpoint/evaluation work but
excludes initialization and the final checkpoint write. `perf_counter` measures
active monotonic elapsed time; OS sleep/wall-clock changes can make timestamp
differences larger. CSV logs have matching JSON metadata with the full config.

Full validation visits every target once with non-overlapping input windows.
Each window receives its own position indices and preceding context within that
window. The small last window is token-weighted. Context ablations therefore
also change evaluation context; this is documented rather than hidden.
