# Public executable: fasttree_gpu. Original FastTree CLI, automatic GPU routing.
.NOTPARALLEL:
.DEFAULT_GOAL := all

CC ?= cc
NVCC ?= nvcc
PYTHON ?= python3
BUILD ?= build
PREFIX ?= /usr/local
DESTDIR ?=
INSTALL ?= install
CUDA_HOST_COMPILER ?=

# Choose CUDA when its compiler is installed, otherwise build the CPU fallback.
# Explicit make cpu / make cuda overrides this choice.
DEFAULT_BACKEND ?= $(shell command -v $(firstword $(NVCC)) >/dev/null 2>&1 && echo cuda || echo cpu)
ifeq ($(filter $(DEFAULT_BACKEND),cpu cuda),)
$(error DEFAULT_BACKEND must be cpu or cuda)
endif

CFLAGS_REF := -O3 -std=gnu11 -march=native -fno-fast-math -ffp-contract=off
CFLAGS_CPU := $(CFLAGS_REF) -Wall
CFLAGS_TEST := -O2 -std=gnu11
NVCC_FMAD ?= false
NVCCFLAGS ?= -O3 -std=c++17 -Xcompiler -Wall -fmad=$(NVCC_FMAD)
ifneq ($(strip $(CUDA_HOST_COMPILER)),)
NVCCFLAGS += -ccbin $(CUDA_HOST_COMPILER)
endif
CUDA_ARCH ?= $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | sed 's/^/sm_/')
ifeq ($(strip $(CUDA_ARCH)),)
CUDA_ARCH := native
endif
ifeq ($(CUDA_ARCH),sm_)
CUDA_ARCH := native
endif

INCLUDES := -Igpu
SRC := src/FastTree.c
BACKEND := gpu/ft_gpu_backend.c
PACK := gpu/profile_pack.c
STUB := gpu/ft_gpu_stub.c
UPSTREAM := upstream/FastTree-2.2.0.c

.PHONY: all cpu cuda reference reference-opt reference-mp tests tests-cuda bench-cuda check check-cuda install clean dist help
all: $(DEFAULT_BACKEND)

$(BUILD):
	@mkdir -p "$(BUILD)"

# Legacy developer binary names are retained inside BUILD for old local evidence
# scripts. Both build modes also produce BUILD/fasttree_gpu and ./fasttree_gpu.
cpu: $(BUILD)
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(SRC) -o "$(BUILD)/FastTree.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(BACKEND) -o "$(BUILD)/ft_gpu_backend.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(PACK) -o "$(BUILD)/profile_pack.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(STUB) -o "$(BUILD)/ft_gpu_stub.o"
	$(CC) "$(BUILD)/FastTree.o" "$(BUILD)/ft_gpu_backend.o" "$(BUILD)/profile_pack.o" "$(BUILD)/ft_gpu_stub.o" -lm -o "$(BUILD)/FastTree-cpu"
	cp "$(BUILD)/FastTree-cpu" "$(BUILD)/fasttree_gpu"
	cp "$(BUILD)/fasttree_gpu" fasttree_gpu
	@echo "Built fasttree_gpu (CPU fallback)"

cuda: $(BUILD)
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(SRC) -o "$(BUILD)/FastTree.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(BACKEND) -o "$(BUILD)/ft_gpu_backend.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(PACK) -o "$(BUILD)/profile_pack.o"
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $(INCLUDES) -c gpu/ft_gpu_cuda.cu -o "$(BUILD)/ft_gpu_cuda.o"
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) "$(BUILD)/FastTree.o" "$(BUILD)/ft_gpu_backend.o" "$(BUILD)/profile_pack.o" "$(BUILD)/ft_gpu_cuda.o" -lm -o "$(BUILD)/FastTree-cuda"
	cp "$(BUILD)/FastTree-cuda" "$(BUILD)/fasttree_gpu"
	cp "$(BUILD)/fasttree_gpu" fasttree_gpu
	@echo "Built fasttree_gpu (CUDA, automatic CPU fallback)"

reference: $(BUILD)
	$(CC) $(CFLAGS_REF) $(UPSTREAM) -lm -o "$(BUILD)/FastTreeCPU-ref"
reference-opt: $(BUILD)
	$(CC) -O3 -std=gnu11 -march=native -funroll-loops $(UPSTREAM) -lm -o "$(BUILD)/FastTreeCPU-opt"
reference-mp: $(BUILD)
	$(CC) $(CFLAGS_REF) -fopenmp -DOPENMP $(UPSTREAM) -lm -o "$(BUILD)/FastTreeMP"

tests: $(BUILD)
	$(CC) $(CFLAGS_TEST) $(INCLUDES) -c tests/test_kernels.c -o "$(BUILD)/test_kernels.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(BACKEND) -o "$(BUILD)/ft_gpu_backend.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(STUB) -o "$(BUILD)/ft_gpu_stub.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(PACK) -o "$(BUILD)/profile_pack.o"
	$(CC) "$(BUILD)/test_kernels.o" "$(BUILD)/ft_gpu_backend.o" "$(BUILD)/ft_gpu_stub.o" "$(BUILD)/profile_pack.o" -lm -o "$(BUILD)/test_kernels"

tests-cuda: $(BUILD)
	$(CC) $(CFLAGS_TEST) $(INCLUDES) -c tests/test_kernels.c -o "$(BUILD)/test_kernels.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(BACKEND) -o "$(BUILD)/ft_gpu_backend.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(PACK) -o "$(BUILD)/profile_pack.o"
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $(INCLUDES) -c gpu/ft_gpu_cuda.cu -o "$(BUILD)/ft_gpu_cuda.o"
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) "$(BUILD)/test_kernels.o" "$(BUILD)/ft_gpu_backend.o" "$(BUILD)/profile_pack.o" "$(BUILD)/ft_gpu_cuda.o" -lm -o "$(BUILD)/test_kernels-cuda"

bench-cuda: $(BUILD)
	$(CC) -O2 -std=gnu11 $(INCLUDES) -c bench/bench_kernels.c -o "$(BUILD)/bench_kernels.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(BACKEND) -o "$(BUILD)/ft_gpu_backend.o"
	$(CC) $(CFLAGS_CPU) $(INCLUDES) -c $(PACK) -o "$(BUILD)/profile_pack.o"
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $(INCLUDES) -c gpu/ft_gpu_cuda.cu -o "$(BUILD)/ft_gpu_cuda.o"
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) "$(BUILD)/bench_kernels.o" "$(BUILD)/ft_gpu_backend.o" "$(BUILD)/profile_pack.o" "$(BUILD)/ft_gpu_cuda.o" -lm -o "$(BUILD)/bench_kernels"

check: reference cpu tests
	"$(BUILD)/test_kernels" --nt --alignment tests/data/nt24.fa
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) tests/check_cli.py --binary "$(BUILD)/fasttree_gpu" --reference "$(BUILD)/FastTreeCPU-ref"

check-cuda: reference cuda tests-cuda
	"$(BUILD)/test_kernels-cuda" --require-gpu --nt --alignment tests/data/nt24.fa
	"$(BUILD)/test_kernels-cuda" --require-gpu --alignment tests/data/aa24.fa
	$(PYTHON) -m unittest discover -s tests -p 'test_*.py'
	$(PYTHON) tests/check_cli.py --binary "$(BUILD)/fasttree_gpu" --reference "$(BUILD)/FastTreeCPU-ref"

install: all
	$(INSTALL) -d "$(DESTDIR)$(PREFIX)/bin"
	$(INSTALL) -m 755 "$(BUILD)/fasttree_gpu" "$(DESTDIR)$(PREFIX)/bin/fasttree_gpu"

clean:
	rm -rf "$(BUILD)"
	rm -f fasttree_gpu

dist:
	$(PYTHON) scripts/package_release.py

help:
	@echo 'make                 Build fasttree_gpu (CUDA if nvcc exists, else CPU)'
	@echo 'make cuda|cpu        Explicit build; CUDA_ARCH=sm_86 for RTX A5000'
	@echo 'make install         Install to PREFIX/bin (default /usr/local/bin)'
	@echo 'make check           CPU, CLI compatibility and tooling tests'
	@echo 'make check-cuda      GPU-required numerical and CLI tests'
	@echo 'make dist            Curated source archive in dist/'
