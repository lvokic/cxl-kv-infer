# HAKES submodule

- Upstream: <https://github.com/nusdbsystem/HAKES-Search>
- Pinned commit: `9fbe8b99371c70d854618cff6de60e9986f4bd95`
- Original source archive SHA-256: `8de74f1e40db97ed89b78604d26ceb8f7924eee2e19187fc2540849fc93a7414`
- License: Apache-2.0 for HAKES; see `faiss-ex/LICENSE` in the submodule for its Faiss-derived code.

The outer repository records HAKES as a Git submodule at `third_party/hakes`; it does not
track the upstream source files directly. Initialize it with:

```bash
git submodule update --init third_party/hakes
```

This is the selected CPU IVF-PQ FastScan implementation for the current indexing direction.
Faiss CPU remains the explicit serialized control. The project `CandidateIndex` adapter and
index-lifecycle integration are still pending, so the model runner does not yet execute HAKES
and does not silently fall back to Faiss.

The independent SIFT1M CPU exploration built and ran this pinned upstream implementation with
temporary harness changes isolated to its run directory. It is an exploratory ANN result, not
this repository's model/index adapter. The local report is
`runs/hakes-sift1m-cpu16-20261015T024344Z/RESULTS.md` (ignored run artifact).

The `libuv` and `llhttp` entries in the upstream `.gitmodules` are for its optional serving
stack; the index-library path does not need those submodules. Follow the upstream `README.md`
for its CPU-only Faiss-derived library build prerequisites. Do not link its modified `faiss-ex`
interchangeably with the repository's PyPI `faiss-cpu` wheel.
