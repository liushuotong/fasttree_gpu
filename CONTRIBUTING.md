# Contributing

Preserve FastTree command-line behavior, model settings, search order, and
upstream notices. Numerical changes need output comparisons against the
reference build and CUDA differential tests on a GPU host. Keep FP64 and the
default compiler arithmetic flags.

Run `make check` and, for CUDA changes, `make check-cuda`. Describe the problem,
change, and validation in pull requests. Retain raw outputs for mismatches.

Add new public files to `release-files.txt`, then run:

```sh
python3 scripts/package_release.py --update-ignore
python3 scripts/package_release.py --check
make dist
```

The manifest controls both Git inclusion and the source archive. Build outputs,
local datasets, and benchmark results are excluded. Contributions are licensed
under GPL version 3 or later, consistent with this distribution.
