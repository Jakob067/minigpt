# v0.1.0 · Python implementation

- Implement a decoder-only GPT using manual causal attention, pre-LayerNorm,
  residual connections, GELU MLPs, learned positions, tied weights, and GPT-2 initialization.
- Add validated YAML settings, character tokenization, checksum-verified data,
  reproducible batch sampling, CUDA/MPS/CPU selection, and locked dependencies.
- Add AdamW, warmup/cosine LR, gradient clipping, full-state checkpoints,
  exact CPU resume tests, evaluation, baselines, and seeded sampling.
- Complete a 5,000-step Small training run and twelve equal-token ablations.
  Full validation loss: 1.4531 nats; perplexity: 4.2763; bits/char: 2.0964.
- Include raw reports, plots, unedited samples, the small Tiny checkpoint,
  architecture notes, and German interview preparation.
- Configure GitHub Actions; remote CI is pending the first user-managed push.
  No remote was pushed and no optional Rust/web/export component was added.
