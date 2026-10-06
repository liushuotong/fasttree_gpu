# Optional GPU controls

Ordinary usage is `fasttree_gpu [FastTree options] alignment > tree`; all GPU
controls below are optional. Use `fasttree_gpu -help` or `-expert` for the
inherited FastTree options. Nucleotides require `-nt`; default is protein.

| Option | Meaning |
| --- | --- |
| `-backend auto` | Default: lazy initialization and work-threshold routing, with CPU fallback. |
| `-backend cpu` | CPU numerical path, including in a CUDA-linked executable. |
| `-backend cuda` | Initialize/require a usable device. Unsupported or small operations can still run on CPU. |
| `-gpu-device ID` | Select the device (default device 0). |
| `-gpu-memory-mb N` | Device cache and workspace budget; 0 means no explicit budget. |
| `-gpu-stats` | Print device work, transfers, fallback and resident/search counters to stderr. |
| `-gpu-min-work N` | Batch work threshold; default 200,000 site-pairs. |
| `-gpu-min-pairs N` | Minimum batch pair count. |
| `-gpu-verify` | Ordinary batch tolerance checks; every resident trial is checked bitwise. Adds CPU work. |
| `-gpu-search-verify` | Experimental JC posterior, quartet likelihood and NNI/SPR distance offload; every candidate requires exact CPU equality. Slow. |

The search experiment disables the normal batch/resident integrations to
isolate its strict acceptance policy. It does not keep the whole quartet
chain resident on device. Differences or unsupported operations retain CPU
results. Unsupported rate counts (above 64), memory limits and small batches
may use CPU paths. OpenMP is not integrated with the CUDA backend.

Default resident phases check the first trial bitwise. A mismatch disables
that phase and retains CPU results; ordinary batches still have tolerance-based
numerical contracts. Do not claim that default mode universally guarantees
all intermediate FP64 values match on every platform.

GPU diagnostics go to stderr and do not contaminate the Newick stdout stream.
