# Measured throughput

Apple M5 Pro, 24 GiB unified memory, macOS 26.7.2, Python 3.11.18,
PyTorch 2.15.0, FP32 on MPS, four PyTorch CPU threads.

Command: `.venv/bin/python -m minigpt.benchmark`
Five warmup updates followed by twenty measured updates; synchronize before and
after timing. Includes batch sampling, forward, backward, clipping and AdamW.
Excludes process/device initialization, data download, and checkpoint writes.

| Config | Batch | Context | Tokens/s | ms/update |
|---|---:|---:|---:|---:|
| tiny | 32 | 64 | 695,425 | 2.94 |
| small | 16 | 256 | 46,799 | 87.52 |

The benchmark predates the final Small iteration budget: its embedded resolved
config is the original 1,000-step configuration. Architecture and batch size are
unchanged. Raw timings, evaluation overhead, configs, commits and environments
are in [benchmark.json](benchmark.json).

The completed Small run used 5,000 steps / 20,480,000 tokens and recorded
487.35 seconds (8.12 minutes) of monotonic training elapsed time.
The entire 12-run ablation grid, including evaluation and report assembly,
recorded 285.54 seconds (4.76 minutes).
Initialization and OS sleep are not included in these monotonic figures; there
were larger wall-clock timestamp gaps. These are single-machine measurements,
not hardware-independent performance promises. There is no Rust benchmark.
