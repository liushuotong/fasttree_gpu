# Testing

## Build and regression tests

```sh
make check BUILD=build-check
make check-cuda BUILD=build-cuda CUDA_ARCH=sm_86
```

CPU checks compare the public CLI against pristine FastTree, including file
and stdin input, models, `-out`, `-intree`, `-log`, PHYLIP replicates, and matrix
output. GPU cases are skipped in CPU builds. `check-cuda` requires a device.

```sh
build-cuda/test_kernels-cuda --require-gpu --nt --alignment tests/data/nt24.fa
build-cuda/test_kernels-cuda --require-gpu --alignment tests/data/aa24.fa
for tool in memcheck initcheck synccheck racecheck; do
  compute-sanitizer --tool "$tool" --error-exitcode 99 \
    build-cuda/test_kernels-cuda --require-gpu --nt \
    --alignment tests/data/nt24.fa || exit 1
done
```

Use the architecture appropriate for your GPU. Inspect exit status and
passed/failed/skipped counts; sanitizer reports should contain no errors.

## Output comparison

```sh
build-cuda/FastTreeCPU-ref -nt -gtr -seed 12345 tests/data/nt24.fa > cpu.tree
build-cuda/fasttree_gpu -nt -gtr -seed 12345 -backend cuda \
  -gpu-verify -gpu-min-work 0 -gpu-stats tests/data/nt24.fa > gpu.tree
cmp cpu.tree gpu.tree
```

Preserve stderr and check that GPU work actually ran. Test
`-gpu-search-verify` separately. Numerical-tolerance tests and exact tree
comparisons test different contracts. Diagnose mismatches with
`bench/compare_trees.py --tolerance 0`; retain the original outputs.

## Paired benchmarks on Linux

Create `cases.txt`, with one `name|options|alignment` record per line:

```text
jc|-nt|tests/data/nt24.fa
gtr|-nt -gtr|tests/data/nt24.fa
jtt||tests/data/aa24.fa
```

```sh
python3 bench/run_stability.py --build build-cuda --cases cases.txt \
  --out bench/results/run1 --modes cpu,cuda --repeats 5 --cpu 8 --stats
python3 bench/summarize_stability.py bench/results/run1
```

Choose a permitted CPU affinity index; 8 is an example. The output directory
must be new. Use larger aligned inputs for meaningful performance measurements.
Keep the GPU free of other work, alternate CPU/GPU order, and preserve input
and binary hashes, options, logs, and raw times. Verification flags add CPU
work and should be excluded from ordinary timing. The runner fails on process
errors or differing output; the summarizer rejects incomplete or invalid pairs.

Speedup is CPU time / GPU time. Time saved is `100 * (1 - GPU / CPU)`.
Report medians and paired differences. The reported spread is
`(maximum - minimum) / median`, not a confidence interval.
