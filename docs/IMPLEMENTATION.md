# Implementation

`src/FastTree.c` contains the FastTree parser, models, search, and output.
`gpu/ft_gpu.h` exposes a synchronous C ABI; a stub backend supports CPU-only
builds. `upstream/FastTree-2.2.0.c` is the unmodified reference.

Distance batches are integrated at `SetBestHit`, and pair-likelihood batches
at `TreeLogLk`. Device profiles are cached by owner and content serial. Profile
updates and frees invalidate cached entries. Both operands stay pinned during
cache acquisition. Allocations count toward the configured memory budget.

`ft_gpu_resident.cuh` keeps leaf codes, topology, internal vectors, and site
intermediates on the GPU during fixed-tree GTR/model-rate phases. The CPU
builds transition tables and performs final logarithms and sums. JC fixed-tree
and whole-quartet residency are not implemented.

`-gpu-search-verify` experimentally offloads JC posterior construction,
ordered quartet likelihoods, and NNI/SPR distance arithmetic. Each result is
checked against the CPU result with exact equality. Search order and branch
optimization remain on the CPU. This mode adds reference work and is not
intended for timing normal acceleration.

The ABI commits output only after success. Failure leaves caller state intact
for CPU fallback, and C++ exceptions do not cross the ABI. Asynchronous copy
sources remain valid through stream completion. CUDA uses FP64 and
`-fmad=false`; CPU builds disable fast math and FP contraction. Ordinary batch
reductions use numerical tolerances. Resident phases check the first trial
bitwise; `-gpu-verify` checks every trial. Matching output on tested inputs
does not establish universal bitwise equivalence across devices and compilers.
