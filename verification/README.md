# verification

Deterministic agent verification CLI for Kronos. Entry point: `./verify` at repo root.

```bash
./verify seed
./verify run smoke          # or: regression
./verify screenshot chart
./verify check              # exits non-zero if any rule is broken
```

Same inputs produce the same results (fixed seed, SHA256 fixture lock, Matplotlib Agg).
`run regression` needs network and pinned Hugging Face weights; `check` only requires `smoke` + chart baseline.
