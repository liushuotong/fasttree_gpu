# fasttree_gpu

FastTree 2.2.0 with optional NVIDIA CUDA acceleration. The command-line options
and alignment formats follow FastTree; use `fasttree_gpu` as the executable name.

## Build

Requires Make and a C compiler. CUDA builds also require `nvcc`, a compatible
C++ compiler, and an NVIDIA driver. CUDA 12.6 on RTX A5000 has been tested.

```sh
make                         # Select CUDA when nvcc is available, otherwise CPU
make cuda CUDA_ARCH=sm_86    # Explicit CUDA build; choose your GPU architecture
make cpu                    # Build without CUDA
make install PREFIX="$HOME/.local"
```

Add `$HOME/.local/bin` to `PATH` after installation. To install the CPU build,
use `make install DEFAULT_BACKEND=cpu PREFIX="$HOME/.local"`.

## Usage

```sh
./fasttree_gpu protein.fasta > tree.nwk
./fasttree_gpu -nt nucleotide.fasta > tree.nwk
./fasttree_gpu -nt -gtr < nucleotide.fasta > tree.nwk
./fasttree_gpu -nt -out tree.nwk nucleotide.fasta
```

Protein is the default. Use `-nt` for nucleotides and place options before the
input filename. FASTA, interleaved PHYLIP, stdin, and the original FastTree
model and search options are supported. Run `./fasttree_gpu -help` for usage.

The default backend is `auto`. Supported operations use CUDA when the workload
is large enough; other operations use the CPU. GPU flags are optional.
See [GPU options](docs/GPU_OPTIONS.md).

## Implementation and performance

CUDA accelerates distance and likelihood batches, plus resident fixed-tree
model/rate likelihood calculations. Tree-search control remains on the CPU.
The implementation retains FP64 and disables CUDA FMA by default. Parallel
reductions can differ in their last bits; output equivalence must be tested
for the target input and platform.

Historical paired tests on an RTX A5000 showed 20.1% less runtime for a
10,652-sequence GTR+CAT alignment (200.33 s to 160.04 s, five pairs).
JC+CAT savings on the tested size ladder were 3.9–12.1%. Small inputs may gain
little or run faster on the CPU. These measurements are specific to the tested
hardware, inputs, and build.

See [implementation](docs/IMPLEMENTATION.md) and [testing](docs/TESTING.md).

## Tests

```sh
make check
make check-cuda CUDA_ARCH=sm_86
make dist
```

Python 3 is required for testing and packaging. CPU tests run in GitHub Actions;
CUDA tests require an NVIDIA GPU. `make dist` creates a source-only archive.

## License and citation

Derived from [FastTree](https://morgannprice.github.io/fasttree/) by Morgan N.
Price and contributors. Original copyright and license notices are preserved.
FastTree is licensed under GPL version 2 or later; the GPL version 3 text
supplied with the upstream snapshot is included in [LICENSE](LICENSE).
This distribution is available under GPL version 3 or later. See [NOTICE](NOTICE).

Please cite the original method when using this software:
Price MN, Dehal PS, Arkin AP (2010). FastTree 2 — Approximately Maximum-Likelihood
Trees for Large Alignments. PLoS ONE 5(3):e9490.
[doi:10.1371/journal.pone.0009490](https://doi.org/10.1371/journal.pone.0009490).
