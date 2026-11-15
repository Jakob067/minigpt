# Local release verification · 2026-11-15

Verified source commit: `7694b9bcd946966ddbf66fbd15dd9d3bbefa5dd7`.
This record adds documentation only after that checkout was tested.

A new local clone was made without hardlinks into an empty temporary directory.
It had no virtual environment, downloaded dataset, or training outputs. Executed:

```sh
make install
make lint
make test
make train-tiny
.venv/bin/minigpt-generate --checkpoint models/tiny.pt --prompt 'ROMEO:' --max-new-tokens 30 --device cpu
```

All commands exited successfully. Ruff lint/format passed; **32 tests passed in
4.47 seconds**. The default Tiny training downloaded and checksum-verified the
corpus, completed 200 updates on MPS, and saved best/last checkpoints. Generation
from the bundled model succeeded on CPU. `pip check` in the project environment
reported no broken requirements.

The fresh Tiny run's sampled validation loss was **2.458053731918335** versus
**2.4580537557601927** in the original recorded run. Maximum absolute parameter
difference was **0.0003467705682851374**. This confirms the documented boundary:
seeds and identical settings do not guarantee bitwise MPS training equality.
The separate CPU resume test is bit-for-bit; it includes dropout and optimizer
state and is part of the 32-test suite. No cross-device bitwise claim is made.

Bundled Tiny checkpoint SHA-256:
`adaa020831bcdf089484de99a108e0d213169c7c2b548dfe3e846436bcd2ef66`.

GitHub Actions is configured but has **not** been executed remotely. There has
been no push. Linux/CUDA behavior was not tested on this Mac; local CPU and MPS
paths were exercised. Optional Rust/web/export components are absent.
