/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ft_gpu_cuda.cu -- CUDA implementation of the FastTree numerical backend.
 *
 * Design notes (see docs/IMPLEMENTATION.md):
 *  - One device context owns a cache of uploaded profiles keyed by
 *    (owner pointer, serial).  Host profiles are immutable after construction
 *    in FastTree 2.2.0 except for the out-profile, which calls
 *    ft_gpu_profile_invalidate()/a serial bump, so a serial is a sound
 *    content identity.
 *  - Batch APIs are synchronous.  Results are written into device temporaries
 *    and copied out only after the kernel succeeded, so a failure never leaves
 *    half-updated host state.
 *  - All arithmetic is FP64 and mirrors the expressions in FastTree.c
 *    statement by statement; only the summation order of the final reduction
 *    differs (fixed tree reduction, deterministic for a given grid/block).
 */

#include "ft_gpu.h"
#include "profile_pack.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <map>
#include <stdexcept>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

/* ---- numeric constants copied from FastTree.c (FP64 build) ---- */
constexpr double kLkUnderflow = 1.0e-4;
constexpr double kLkUnderflowInv = 1.0e4;
constexpr double kLogLkUnderflow = 9.21034037197618;
constexpr double kMLMinRelBranchLength = 2.5e-9;

constexpr int kThreads = 256;      /* threads per pair for PairLogLk */
constexpr int kDistThreads = 128;  /* threads per pair for distance kernels */
/* above this many sites per pair, one block per pair beats one warp per pair */
constexpr int kWarpKernelMaxSites = 8192;

int64_t now_us() {
  using namespace std::chrono;
  return duration_cast<microseconds>(steady_clock::now().time_since_epoch()).count();
}

struct ProfileDev {
  const double *weights;
  const unsigned char *codes;
  const int *index;
  const double *vectors;
  const double *codeDist;
};

struct DevPair {
  ProfileDev A;
  ProfileDev B;
  double length;
  int wantSite;
};

struct DevDistPair {
  ProfileDev A;
  ProfileDev B;
  const unsigned char *codesA;
  const unsigned char *codesB;
  int isSeq;
};

/* ---------------------------------------------------------------- kernels */

/* PairLogLk, matrix model (GTR / protein).  One block per pair; each thread
   walks a block-strided subset of sites accumulating a running product with
   the same underflow bookkeeping as the CPU loop, and the per-thread
   log-contributions are reduced in a fixed tree. */
template <int NC>
__global__ void pair_loglk_matrix_kernel(
    const DevPair *__restrict__ pairs,
    const unsigned int *__restrict__ ratecat, int nPos,
    const double *__restrict__ rates, int nRateCats,
    const double *__restrict__ eigenval,
    const double *__restrict__ codeFreq,
    double *__restrict__ out_loglk,
    double *__restrict__ out_site /* [nPairs*nPos] or NULL */,
    const double *transitions) {
  extern __shared__ double smem[];
  const int pairIdx = blockIdx.x;
  const DevPair p = pairs[pairIdx];
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  /* exp(eigenvalue * relative length) per rate category */
  for (int r = tid; r < nRateCats; r += nt) {
    double relLen = p.length * rates[r];
    if (relLen < kMLMinRelBranchLength) relLen = kMLMinRelBranchLength;
    double *dst = &smem[r * NC];
#pragma unroll
    for (int j = 0; j < NC; j++) dst[j] = transitions ? transitions[((size_t)pairIdx*nRateCats+r)*NC+j] : exp(relLen * eigenval[j]);
  }
  __syncthreads();

  double lk = 1.0;
  double loglk = 0.0;
  double *siteRow = (out_site != NULL) ? &out_site[(size_t)pairIdx * nPos] : NULL;

  for (int i = tid; i < nPos; i += nt) {
    const int iRate = (int)ratecat[i];
    const double *eig = &smem[iRate * NC];
    const double wA = p.A.weights[i];
    const double wB = p.B.weights[i];
    const unsigned int codeA = p.A.codes[i];
    const unsigned int codeB = p.B.codes[i];
    if (wA == 0.0 && wB == 0.0 && codeA == FT_GPU_NOCODE && codeB == FT_GPU_NOCODE) {
      /* Likelihood of A vs B is 1: nothing changes, and the CPU loop also
         skips the vector advance for this site. */
      if (siteRow != NULL) siteRow[i] = 1.0;
      continue;
    }
    const int idxA = p.A.index[i];
    const int idxB = p.B.index[i];
    const double *FA = (idxA >= 0) ? &p.A.vectors[(size_t)idxA * NC]
                                   : &codeFreq[(size_t)codeA * FT_GPU_CODE_STRIDE];
    const double *FB = (idxB >= 0) ? &p.B.vectors[(size_t)idxB * NC]
                                   : &codeFreq[(size_t)codeB * FT_GPU_CODE_STRIDE];
    const bool mixA = (wA > 0.0 && wA < 1.0);
    const bool mixB = (wB > 0.0 && wB < 1.0);
    double lkAB = 0.0;
    if (!mixA && !mixB) {
#pragma unroll
      for (int j = 0; j < NC; j++) lkAB += eig[j] * FA[j] * FB[j];
    } else if (mixA && !mixB) {
      const double *gap = &codeFreq[(size_t)FT_GPU_NOCODE * FT_GPU_CODE_STRIDE];
#pragma unroll
      for (int j = 0; j < NC; j++) {
        const double a = wA * FA[j] + (1.0 - wA) * gap[j];
        lkAB += eig[j] * a * FB[j];
      }
    } else if (!mixA && mixB) {
      const double *gap = &codeFreq[(size_t)FT_GPU_NOCODE * FT_GPU_CODE_STRIDE];
#pragma unroll
      for (int j = 0; j < NC; j++) {
        const double b = wB * FB[j] + (1.0 - wB) * gap[j];
        lkAB += eig[j] * FA[j] * b;
      }
    } else {
      const double *gap = &codeFreq[(size_t)FT_GPU_NOCODE * FT_GPU_CODE_STRIDE];
#pragma unroll
      for (int j = 0; j < NC; j++) {
        const double a = wA * FA[j] + (1.0 - wA) * gap[j];
        const double b = wB * FB[j] + (1.0 - wB) * gap[j];
        lkAB += eig[j] * a * b;
      }
    }
    if (siteRow != NULL) siteRow[i] = lkAB;
    if (transitions) continue; /* host performs the original ordered reduction */
    if (!(lkAB > 0.0) || !isfinite(lkAB)) { lk = NAN; continue; }
    lk *= lkAB;
    if (!(lk > 0.0) || !isfinite(lk)) { lk = NAN; continue; }
    while (lk < kLkUnderflow) {
      lk *= kLkUnderflowInv;
      loglk -= kLogLkUnderflow;
    }
    while (lk > kLkUnderflowInv) {
      lk *= kLkUnderflow;
      loglk += kLogLkUnderflow;
    }
  }
  loglk += log(lk);

  /* fixed-order block reduction */
  __shared__ double red[kThreads];
  red[tid] = loglk;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) out_loglk[pairIdx] = red[0];
}

/* PairLogLk, Jukes-Cantor (transmat == NULL, nCodes == 4). */
__global__ void pair_loglk_jc_kernel(
    const DevPair *__restrict__ pairs,
    const unsigned int *__restrict__ ratecat, int nPos,
    const double *__restrict__ rates, int nRateCats,
    double *__restrict__ out_loglk,
    double *__restrict__ out_site, const double *transitions) {
  extern __shared__ double smem[];
  double *pSameS = smem;
  double *pDiffS = smem + nRateCats;
  const int pairIdx = blockIdx.x;
  const DevPair p = pairs[pairIdx];
  const int tid = threadIdx.x;
  const int nt = blockDim.x;

  for (int r = tid; r < nRateCats; r += nt) {
    pSameS[r] = transitions ? transitions[(size_t)pairIdx*2*nRateCats+r] : 0.25 + 0.75 * exp((-4.0 / 3.0) * fabs(p.length * rates[r]));
    pDiffS[r] = transitions ? transitions[((size_t)pairIdx*2+1)*nRateCats+r] : (1.0 - pSameS[r]) / 3.0;
  }
  __syncthreads();

  double lk = 1.0;
  double loglk = 0.0;
  double *siteRow = (out_site != NULL) ? &out_site[(size_t)pairIdx * nPos] : NULL;

  for (int i = tid; i < nPos; i += nt) {
    const int iRate = (int)ratecat[i];
    const double pSame = pSameS[iRate];
    const double pDiff = pDiffS[iRate];
    const double wA = p.A.weights[i];
    const double wB = p.B.weights[i];
    const int codeA = (int)p.A.codes[i];
    const int codeB = (int)p.B.codes[i];
    const int idxA = p.A.index[i];
    const int idxB = p.B.index[i];
    const double *fA = (idxA >= 0) ? &p.A.vectors[(size_t)idxA * 4] : NULL;
    const double *fB = (idxB >= 0) ? &p.B.vectors[(size_t)idxB * 4] : NULL;
    double lkAB;
    if (fA == NULL && fB == NULL) {
      if (codeA == FT_GPU_NOCODE) {
        lkAB = 0.25;
      } else if (codeB == FT_GPU_NOCODE) {
        lkAB = 0.25;
      } else if (codeA == codeB) {
        lkAB = pSame * wA * wB + 0.25 * (1 - wA * wB);
      } else {
        lkAB = pDiff * wA * wB + 0.25 * (1 - wA * wB);
      }
    } else if (fA == NULL) {
      if (codeA == FT_GPU_NOCODE) {
        lkAB = 0.25;
      } else {
        lkAB = wA * (pDiff + fB[codeA] * (pSame - pDiff)) + (1.0 - wA) * 0.25;
      }
    } else if (fB == NULL) {
      if (codeB == FT_GPU_NOCODE) {
        lkAB = 0.25;
      } else {
        lkAB = wB * (pDiff + fA[codeB] * (pSame - pDiff)) + (1.0 - wB) * 0.25;
      }
    } else {
      lkAB = 0.0;
#pragma unroll
      for (int j = 0; j < 4; j++)
        lkAB += fB[j] * (fA[j] * pSame + (1 - fA[j]) * pDiff);
    }
    if (siteRow != NULL) siteRow[i] = lkAB;
    if (transitions) continue; /* host performs the original ordered reduction */
    if (!(lkAB > 0.0) || !isfinite(lkAB)) { lk = NAN; continue; }
    lk *= lkAB;
    if (!(lk > 0.0) || !isfinite(lk)) { lk = NAN; continue; }
    while (lk < kLkUnderflow) {
      lk *= kLkUnderflowInv;
      loglk -= kLogLkUnderflow;
    }
  }
  /* JC branch in FastTree.c has no upper-rescale loop */
  loglk += log(lk);

  __shared__ double red[kThreads];
  red[tid] = loglk;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) red[tid] += red[tid + s];
    __syncthreads();
  }
  if (tid == 0) out_loglk[pairIdx] = red[0];
}

__device__ __forceinline__ double dist_piece(
    unsigned int code1, unsigned int code2, const double *f1, const double *f2,
    int useMatrix, const double *dmatDist, const double *dmatEigenval,
    const double *dmatCodeFreq, const double *codeDist2, int nCodes) {
  if (useMatrix) {
    if (code1 != FT_GPU_NOCODE && code2 != FT_GPU_NOCODE) {
      return dmatDist[(size_t)code1 * FT_GPU_MAXCODES + code2];
    } else if (codeDist2 != NULL && code1 != FT_GPU_NOCODE) {
      return codeDist2[code1];
    } else {
      const double *a = f1;
      const double *b = f2;
      if (a == NULL) {
        if (code1 == FT_GPU_NOCODE) return 10.0;
        a = &dmatCodeFreq[(size_t)code1 * FT_GPU_MAXCODES];
      }
      if (b == NULL) {
        if (code2 == FT_GPU_NOCODE) return 10.0;
        b = &dmatCodeFreq[(size_t)code2 * FT_GPU_MAXCODES];
      }
      double sum = 0.0;
      for (int k = 0; k < nCodes; k++) sum += a[k] * b[k] * dmatEigenval[k];
      return sum;
    }
  } else {
    if (code1 != FT_GPU_NOCODE) {
      if (code2 != FT_GPU_NOCODE) {
        return (code1 == code2) ? 0.0 : 1.0;
      } else {
        if (f2 == NULL) return 10.0;
        return 1.0 - f2[code1];
      }
    } else {
      if (code2 != FT_GPU_NOCODE) {
        if (f1 == NULL) return 10.0;
        return 1.0 - f1[code2];
      } else {
        if (f1 == NULL || f2 == NULL) return 10.0;
        double piece = 1.0;
        for (int k = 0; k < nCodes; k++) piece -= f1[k] * f2[k];
        return piece;
      }
    }
  }
}

/* Independent per-site work followed by the CPU's exact site order. */
__global__ void dist_ordered_sites(const DevDistPair *pairs,int nPos,int nCodes,int useMatrix,
 const double *dmatDist,const double *dmatEigenval,const double *dmatCodeFreq,double *values) {
  int pair=blockIdx.x,site=blockIdx.y*blockDim.x+threadIdx.x;
  if(site>=nPos)return;
  const DevDistPair p=pairs[pair];
  size_t index=((size_t)pair*nPos+site)*2;
  double wa=p.A.weights[site],wb=p.B.weights[site];
  values[index]=0.0;values[index+1]=0.0;
  if(wa>0.0&&wb>0.0) {
    const double *a=p.A.index[site]>=0?p.A.vectors+(size_t)p.A.index[site]*nCodes:NULL;
    const double *b=p.B.index[site]>=0?p.B.vectors+(size_t)p.B.index[site]*nCodes:NULL;
    const double *cd=p.B.codeDist?p.B.codeDist+(size_t)site*nCodes:NULL;
    double weight=wa*wb;
    double piece=dist_piece(p.A.codes[site],p.B.codes[site],a,b,useMatrix,dmatDist,dmatEigenval,dmatCodeFreq,cd,nCodes);
    values[index]=weight*piece;values[index+1]=weight;
  }
}
__global__ void dist_ordered_reduce(const DevDistPair *pairs,int nPos,const double *values,double *dist,double *weight) {
  int pair=blockIdx.x;double top=0.0,denom=0.0;
  for(int site=0;site<nPos;site++) {
    if(pairs[pair].A.weights[site]>0.0&&pairs[pair].B.weights[site]>0.0) {
      size_t index=((size_t)pair*nPos+site)*2;
      denom+=values[index+1];top+=values[index];
    }
  }
  weight[pair]=denom>0?denom:0.01;dist[pair]=denom>0?top/denom:1.0;
}

/* ProfileDist (raw dist/weight) and SeqDist in one kernel. */
__global__ void dist_kernel(
    const DevDistPair *__restrict__ pairs, int nPos, int nCodes, int useMatrix,
    const double *__restrict__ dmatDist,
    const double *__restrict__ dmatEigenval,
    const double *__restrict__ dmatCodeFreq,
    double *__restrict__ outDist, double *__restrict__ outWeight) {
  __shared__ double redTop[kDistThreads];
  __shared__ double redDen[kDistThreads];
  const int pairIdx = blockIdx.x;
  const DevDistPair p = pairs[pairIdx];
  const unsigned char *codesA = (p.codesA != NULL) ? p.codesA : p.A.codes;
  const unsigned char *codesB = (p.codesB != NULL) ? p.codesB : p.B.codes;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  double top = 0.0;
  double denom = 0.0;

  if (p.isSeq) {
    int nUse = 0;
    if (!useMatrix) {
      int nDiff = 0;
      for (int i = tid; i < nPos; i += nt) {
        const unsigned char c1 = codesA[i];
        const unsigned char c2 = codesB[i];
        if (c1 != FT_GPU_NOCODE && c2 != FT_GPU_NOCODE) {
          nUse++;
          if (c1 != c2) nDiff++;
        }
      }
      top = (double)nDiff;
    } else {
      for (int i = tid; i < nPos; i += nt) {
        const unsigned char c1 = codesA[i];
        const unsigned char c2 = codesB[i];
        if (c1 != FT_GPU_NOCODE && c2 != FT_GPU_NOCODE) {
          nUse++;
          top += dmatDist[(size_t)c1 * FT_GPU_MAXCODES + c2];
        }
      }
    }
    redTop[tid] = top;
    redDen[tid] = (double)nUse;
    __syncthreads();
    for (int s = nt / 2; s > 0; s >>= 1) {
      if (tid < s) {
        redTop[tid] += redTop[tid + s];
        redDen[tid] += redDen[tid + s];
      }
      __syncthreads();
    }
    if (tid == 0) {
      const double weight = redDen[0];
      outWeight[pairIdx] = weight;
      outDist[pairIdx] = (weight > 0.0) ? redTop[0] / weight : 1.0;
    }
    return;
  }

  for (int i = tid; i < nPos; i += nt) {
    const double w1 = p.A.weights[i];
    const double w2 = p.B.weights[i];
    if (w1 > 0.0 && w2 > 0.0) {
      const double weight = w1 * w2;
      denom += weight;
      const int idx1 = p.A.index[i];
      const int idx2 = p.B.index[i];
      const double *f1 = (idx1 >= 0) ? &p.A.vectors[(size_t)idx1 * nCodes] : NULL;
      const double *f2 = (idx2 >= 0) ? &p.B.vectors[(size_t)idx2 * nCodes] : NULL;
      const double *codeDist2 =
          (p.B.codeDist != NULL) ? &p.B.codeDist[(size_t)i * nCodes] : NULL;
      const double piece = dist_piece(p.A.codes[i], p.B.codes[i], f1, f2, useMatrix,
                                      dmatDist, dmatEigenval, dmatCodeFreq,
                                      codeDist2, nCodes);
      top += weight * piece;
    }
  }
  redTop[tid] = top;
  redDen[tid] = denom;
  __syncthreads();
  for (int s = nt / 2; s > 0; s >>= 1) {
    if (tid < s) {
      redTop[tid] += redTop[tid + s];
      redDen[tid] += redDen[tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    const double d = redDen[0];
    outWeight[pairIdx] = (d > 0.0) ? d : 0.01;
    outDist[pairIdx] = (d > 0.0) ? redTop[0] / d : 1.0;
  }
}

/* Warp-per-pair variant: one warp per distance job, no block-wide barriers.
   For short alignments (the common case: hundreds of pairs of a few hundred to
   a few thousand sites) this removes most of the per-pair overhead of the
   block-per-pair kernel. */
__global__ void dist_warp_kernel(
    const DevDistPair *__restrict__ pairs, int nPairs, int nPos, int nCodes,
    int useMatrix, const double *__restrict__ dmatDist,
    const double *__restrict__ dmatEigenval, const double *__restrict__ dmatCodeFreq,
    double *__restrict__ outDist, double *__restrict__ outWeight) {
  const int warpsPerBlock = blockDim.x >> 5;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int pairIdx = blockIdx.x * warpsPerBlock + warp;
  if (pairIdx >= nPairs) return;
  const DevDistPair p = pairs[pairIdx];
  const unsigned char *codesA = (p.codesA != NULL) ? p.codesA : p.A.codes;
  const unsigned char *codesB = (p.codesB != NULL) ? p.codesB : p.B.codes;

  double top = 0.0;
  double denom = 0.0;
  int nUse = 0;

  if (p.isSeq) {
    if (!useMatrix) {
      int nDiff = 0;
      for (int i = lane; i < nPos; i += 32) {
        const unsigned char c1 = codesA[i];
        const unsigned char c2 = codesB[i];
        if (c1 != FT_GPU_NOCODE && c2 != FT_GPU_NOCODE) {
          nUse++;
          if (c1 != c2) nDiff++;
        }
      }
      top = (double)nDiff;
    } else {
      for (int i = lane; i < nPos; i += 32) {
        const unsigned char c1 = codesA[i];
        const unsigned char c2 = codesB[i];
        if (c1 != FT_GPU_NOCODE && c2 != FT_GPU_NOCODE) {
          nUse++;
          top += dmatDist[(size_t)c1 * FT_GPU_MAXCODES + c2];
        }
      }
    }
  } else {
    for (int i = lane; i < nPos; i += 32) {
      const double w1 = p.A.weights[i];
      const double w2 = p.B.weights[i];
      if (w1 > 0.0 && w2 > 0.0) {
        const double weight = w1 * w2;
        denom += weight;
        const int idx1 = p.A.index[i];
        const int idx2 = p.B.index[i];
        const double *f1 = (idx1 >= 0) ? &p.A.vectors[(size_t)idx1 * nCodes] : NULL;
        const double *f2 = (idx2 >= 0) ? &p.B.vectors[(size_t)idx2 * nCodes] : NULL;
        const double *codeDist2 =
            (p.B.codeDist != NULL) ? &p.B.codeDist[(size_t)i * nCodes] : NULL;
        const double piece = dist_piece(p.A.codes[i], p.B.codes[i], f1, f2, useMatrix,
                                        dmatDist, dmatEigenval, dmatCodeFreq, codeDist2, nCodes);
        top += weight * piece;
      }
    }
  }
  /* warp reduction (fixed order) */
  for (int off = 16; off > 0; off >>= 1) {
    top += __shfl_down_sync(0xffffffffu, top, off);
    denom += __shfl_down_sync(0xffffffffu, denom, off);
    nUse += __shfl_down_sync(0xffffffffu, nUse, off);
  }
  if (lane == 0) {
    if (p.isSeq) {
      const double w = (double)nUse;
      outWeight[pairIdx] = w;
      outDist[pairIdx] = (w > 0.0) ? top / w : 1.0;
    } else {
      outWeight[pairIdx] = (denom > 0.0) ? denom : 0.01;
      outDist[pairIdx] = (denom > 0.0) ? top / denom : 1.0;
    }
  }
}

/* ---------------------------------------------------------------- context */

struct Key {
  const void *owner;
  uint64_t serial;
  bool operator==(const Key &o) const {
    return owner == o.owner && serial == o.serial;
  }
};

struct KeyHash {
  size_t operator()(const Key &k) const {
    uint64_t h = (uint64_t)(uintptr_t)k.owner;
    h ^= k.serial + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return (size_t)h;
  }
};

struct UploadBlock { unsigned char *ptr; size_t capacity, used; };

struct Entry {
  Key key;
  int nPos;
  int nCodes;
  int nVectors;
  size_t bytes;
  uint64_t lastUse;
  double *dWeights;
  unsigned char *dCodes;
  int *dIndex;
  double *dVectors;
  double *dCodeDist;
};

void set_error(FtGpuContext *ctx, const char *op, const std::string &msg);
struct BackendFailure { FtGpuStatus status; };
void fail(FtGpuContext *ctx, FtGpuStatus status, const char *op, const char *message);
void cuda_checked(FtGpuContext *ctx, cudaError_t error, const char *op);
bool reserve_bytes(FtGpuContext *ctx, size_t bytes, const char *op);
void release_bytes(FtGpuContext *ctx, size_t bytes);

template <typename T>
struct Buf {
  T *ptr;
  size_t cap;
  Buf() : ptr(NULL), cap(0) {}
  bool ensure(FtGpuContext *ctx, size_t n, const char *what);
  void release(FtGpuContext *ctx) {
    if (ptr) { cudaFree(ptr); release_bytes(ctx, cap * sizeof(T)); }
    ptr = NULL;
    cap = 0;
  }
};

struct Scratch {
  Buf<double> site;      /* per-site likelihood output */
  Buf<double> dist;      /* batch scalars */
  Buf<double> weight;
  Buf<DevPair> pairs;
  Buf<DevDistPair> dpairs;
  Buf<unsigned int> ratecat;
  Buf<double> rates;
  Buf<double> transitions;
  Buf<double> eigenval;
  Buf<double> codeFreq;
  Buf<double> dmatDist;
  Buf<double> dmatCodeFreq;
  Buf<double> dmatEig;
  Buf<unsigned char> seqCodes; /* packed codes for code/code distance jobs */
  std::vector<unsigned char> hostSeqCodes;
  void release(FtGpuContext *ctx) {
    site.release(ctx); dist.release(ctx); weight.release(ctx); pairs.release(ctx); dpairs.release(ctx);
    ratecat.release(ctx); rates.release(ctx); eigenval.release(ctx); codeFreq.release(ctx);
    transitions.release(ctx); dmatDist.release(ctx); dmatCodeFreq.release(ctx); dmatEig.release(ctx); seqCodes.release(ctx);
  }
};

}  // namespace

struct FtGpuContext {
  int deviceId;
  cudaStream_t stream;
  size_t memoryLimit;
  size_t deviceBytes;
  uint64_t clock;
  int verify;
  int statsLevel;
  std::unordered_map<Key, Entry, KeyHash> cache;
  /* Idle storage is reused only without an explicit user memory budget. */
  std::multimap<size_t, double *> idleProfiles;
  size_t idleBytes;
  std::vector<UploadBlock> uploadBlocks;
  size_t pinnedHostBytes;
  Scratch scratch;
  FtGpuStats stats;
  std::string lastError;
  std::string deviceInfo;
  bool fatal;
  int debugSequential;
  cudaEvent_t ev[4];   /* batch timing: start, after H2D, after kernel, after D2H */
  int haveEvents;
  /* per-batch pinned keys (not evicted while in use) */
  std::vector<Key> pinned;
  /* Keep upload sources alive until the synchronous batch boundary. */
  std::vector<std::vector<double>> pendingUploads;
};

namespace {

void release_bytes(FtGpuContext *ctx, size_t bytes) {
  ctx->deviceBytes -= bytes;
  ctx->stats.deviceBytes = ctx->deviceBytes;
}

template <typename T>
bool Buf<T>::ensure(FtGpuContext *ctx, size_t n, const char *what) {
  if (n <= cap) return true;
  if (n > SIZE_MAX / sizeof(T)) fail(ctx, FT_GPU_ERR_OVERFLOW, what, "scratch size overflow");
  release(ctx);
  size_t bytes = n * sizeof(T);
  if (!reserve_bytes(ctx, bytes, what)) return false;
  cudaError_t err = cudaMalloc((void **)&ptr, bytes);
  if (err != cudaSuccess) {
    ptr = NULL;
    cuda_checked(ctx, err, what);
  }
  cap = n;
  ctx->deviceBytes += bytes;
  ctx->stats.deviceBytes = ctx->deviceBytes;
  ctx->stats.peakDeviceBytes = std::max(ctx->stats.peakDeviceBytes, (uint64_t)ctx->deviceBytes);
  return true;
}

bool trace_enabled() {
  static int cached = -1;
  if (cached < 0) {
    const char *v = getenv("FT_GPU_TRACE");
    cached = (v != NULL && v[0] != '0') ? 1 : 0;
  }
  return cached != 0;
}

void set_error(FtGpuContext *ctx, const char *op, const std::string &msg) {
  char buf[FT_GPU_ERROR_MESSAGE_SIZE];
  std::snprintf(buf, sizeof(buf), "%s: %s", op, msg.c_str());
  ctx->lastError = buf;
}

std::string cuda_err(cudaError_t err) {
  return std::string(cudaGetErrorName(err)) + " (" + cudaGetErrorString(err) + ")";
}

void fail(FtGpuContext *ctx, FtGpuStatus status, const char *op, const char *message) {
  set_error(ctx, op, message);
  throw BackendFailure{status};
}

void cuda_checked(FtGpuContext *ctx, cudaError_t err, const char *op) {
  if (err == cudaSuccess) return;
  set_error(ctx, op, cuda_err(err));
  if (err == cudaErrorMemoryAllocation) {
    /* Clear the recoverable error so the next launch is not blamed for it. */
    cudaGetLastError();
    throw BackendFailure{FT_GPU_ERR_MEMORY};
  }
  ctx->fatal = true;
  throw BackendFailure{FT_GPU_ERR_CUDA};
}

void free_entry(FtGpuContext *ctx, Entry &e) {
  if (e.dWeights) cudaFree(e.dWeights);
  release_bytes(ctx, e.bytes);
  e.bytes = 0;
  e.dWeights = NULL;
  e.dCodes = NULL;
  e.dIndex = NULL;
  e.dVectors = NULL;
  e.dCodeDist = NULL;
}

void evict_lru(FtGpuContext *ctx, size_t need) {
  if (ctx->memoryLimit == 0) return;
  while ((need > ctx->memoryLimit || ctx->deviceBytes > ctx->memoryLimit - need) && !ctx->cache.empty()) {
    auto victim = ctx->cache.end();
    uint64_t best = UINT64_MAX;
    for (auto it = ctx->cache.begin(); it != ctx->cache.end(); ++it) {
      bool isPinned = false;
      for (size_t i = 0; i < ctx->pinned.size(); i++) {
        if (ctx->pinned[i] == it->first) {
          isPinned = true;
          break;
        }
      }
      if (isPinned) continue;
      if (it->second.lastUse < best) {
        best = it->second.lastUse;
        victim = it;
      }
    }
    if (victim == ctx->cache.end()) break; /* everything pinned */
    ctx->stats.nProfileEvictions++;
    free_entry(ctx, victim->second);
    ctx->cache.erase(victim);
  }
}

bool reserve_bytes(FtGpuContext *ctx, size_t need, const char *op) {
  if (need > SIZE_MAX - ctx->deviceBytes)
    fail(ctx, FT_GPU_ERR_OVERFLOW, op, "device byte count overflow");
  evict_lru(ctx, need);
  if (ctx->memoryLimit != 0 &&
      (need > ctx->memoryLimit || ctx->deviceBytes > ctx->memoryLimit - need)) {
    set_error(ctx, op, "device memory budget exceeded by pinned profiles and scratch");
    return false;
  }
  return true;
}

bool valid_codes(const unsigned char *codes, int nPos, int nCodes) {
  if (codes == NULL && nPos != 0) return false;
  for (int i = 0; i < nPos; ++i)
    if (codes[i] >= nCodes && codes[i] != FT_GPU_NOCODE) return false;
  return true;
}

/* Sources remain stable until the synchronous batch boundary. Keep at most
   64 MiB pinned; unusually large batches use ordinary host storage beyond it. */
unsigned char *upload_storage(FtGpuContext *ctx, size_t bytes) {
  size_t aligned = (bytes + 7u) & ~size_t(7u);
  for (auto &block : ctx->uploadBlocks) {
    if (aligned <= block.capacity - block.used) {
      unsigned char *result = block.ptr + block.used;
      block.used += aligned;
      return result;
    }
  }
  const size_t limit = 64u * 1024u * 1024u;
  size_t capacity = std::max(aligned, size_t(4u * 1024u * 1024u));
  if (capacity <= limit - ctx->pinnedHostBytes) {
    unsigned char *ptr = NULL;
    cudaError_t status = cudaHostAlloc((void **)&ptr, capacity, cudaHostAllocDefault);
    if (status == cudaSuccess) {
      try { ctx->uploadBlocks.push_back(UploadBlock{ptr, capacity, aligned}); }
      catch (...) { cudaFreeHost(ptr); throw; }
      ctx->pinnedHostBytes += capacity;
      return ptr;
    }
    cuda_checked(ctx, status == cudaErrorMemoryAllocation ? cudaSuccess : status, "host staging");
    cudaGetLastError(); /* pinned memory is optional */
  }
  ctx->pendingUploads.emplace_back((bytes + sizeof(double) - 1) / sizeof(double));
  return reinterpret_cast<unsigned char *>(ctx->pendingUploads.back().data());
}

void reset_uploads(FtGpuContext *ctx) {
  ctx->pendingUploads.clear();
  for (auto &block : ctx->uploadBlocks) block.used = 0;
}

/* Upload a profile if needed; returns the entry or NULL on error. */
const Entry *get_profile(FtGpuContext *ctx, const FtGpuProfileView &view, int nPos,
                         int nCodes, const char *op) {
  Key key;
  key.owner = view.owner;
  key.serial = view.serial;
  auto it = ctx->cache.find(key);
  if (it != ctx->cache.end()) {
    if (it->second.nPos != nPos || it->second.nCodes != nCodes ||
        it->second.nVectors != view.nVectors)
      fail(ctx, FT_GPU_ERR_INVALID, op, "cached profile shape changed without a new serial");
    it->second.lastUse = ++ctx->clock;
    ctx->stats.nProfileHits++;
    return &it->second;
  }

  if (!valid_codes(view.codes, nPos, nCodes))
    fail(ctx, FT_GPU_ERR_INVALID, op, "invalid profile character code");
  for (int i = 0; i < nPos; ++i)
    if (!std::isfinite(view.weights[i]) || view.weights[i] < 0.0)
      fail(ctx, FT_GPU_ERR_INVALID, op, "invalid profile weight");
  if (view.nVectors > nPos || (view.nVectors > 0 && view.vectors == NULL))
    fail(ctx, FT_GPU_ERR_INVALID, op, "invalid profile vectors");
  for (size_t i = 0; i < (size_t)view.nVectors * nCodes; ++i)
    if (!std::isfinite(view.vectors[i]))
      fail(ctx, FT_GPU_ERR_INVALID, op, "non-finite profile vector");

  /* build the vector index exactly like GET_FREQ: a site contributes a vector
     iff weights[site] > 0 && codes[site] == NOCODE */
  std::vector<int> index((size_t)nPos, -1);
  int nVec = 0;
  if (ft_pack_vector_index(view.weights, view.codes, nPos, index.data(), &nVec) != 0) {
    fail(ctx, FT_GPU_ERR_INVALID, op, "could not build the profile vector index");
  }
  if (nVec != view.nVectors) {
    char msg[256];
    std::snprintf(msg, sizeof(msg),
                  "profile vector index mismatch: built %d vectors, nVectors=%d",
                  nVec, view.nVectors);
    fail(ctx, FT_GPU_ERR_INVALID, op, msg);
  }

  size_t bytes = (size_t)nPos * (sizeof(double) + sizeof(unsigned char) + sizeof(int));
  size_t vecBytes = 0;
  if (nVec > 0) {
    if (view.vectors == NULL) {
      set_error(ctx, op, "profile has nVectors > 0 but vectors == NULL");
      return NULL;
    }
    vecBytes = (size_t)nVec * nCodes * sizeof(double);
    bytes += vecBytes;
  }
  size_t cdBytes = 0;
  if (view.codeDist != NULL) {
    cdBytes = (size_t)nPos * nCodes * sizeof(double);
    bytes += cdBytes;
  }
  if (!reserve_bytes(ctx, bytes, op)) return NULL;

  Entry e;
  std::memset(&e, 0, sizeof(e));
  e.key = key;
  e.nPos = nPos;
  e.nCodes = nCodes;
  e.nVectors = nVec;
  e.bytes = 0; /* Account storage immediately after successful acquisition. */
  e.lastUse = ++ctx->clock;

  try {
    /* Double-valued regions first, then int indices and byte codes. Every
       typed region is naturally aligned; bytes is the exact device payload. */
    unsigned char *host = upload_storage(ctx, bytes);
    size_t weightsBytes = (size_t)nPos * sizeof(double);
    size_t indexOffset = weightsBytes + vecBytes + cdBytes;
    size_t codesOffset = indexOffset + (size_t)nPos * sizeof(int);
    std::memcpy(host, view.weights, weightsBytes);
    if (vecBytes) std::memcpy(host + weightsBytes, view.vectors, vecBytes);
    if (cdBytes) std::memcpy(host + weightsBytes + vecBytes, view.codeDist, cdBytes);
    std::memcpy(host + indexOffset, index.data(), (size_t)nPos * sizeof(int));
    std::memcpy(host + codesOffset, view.codes, (size_t)nPos);
    /* Publish the source before submitting DMA so all exception paths keep it
       alive until the wrapper synchronizes the stream. */
    auto idle = ctx->idleProfiles.lower_bound(bytes);
    if (idle != ctx->idleProfiles.end() && idle->first - bytes <= bytes / 8) {
      e.dWeights = idle->second;
      e.bytes = idle->first;
      ctx->idleBytes -= e.bytes;
      ctx->idleProfiles.erase(idle);
    } else {
      cuda_checked(ctx, cudaMalloc((void **)&e.dWeights, bytes), op);
      e.bytes = bytes;
      ctx->deviceBytes += bytes;
    }
    unsigned char *device = reinterpret_cast<unsigned char *>(e.dWeights);
    e.dVectors = vecBytes ? reinterpret_cast<double *>(device + weightsBytes) : NULL;
    e.dCodeDist = cdBytes ? reinterpret_cast<double *>(device + weightsBytes + vecBytes) : NULL;
    e.dIndex = reinterpret_cast<int *>(device + indexOffset);
    e.dCodes = device + codesOffset;
    cuda_checked(ctx, cudaMemcpyAsync(device, host, bytes,
                                      cudaMemcpyHostToDevice, ctx->stream), op);
    auto res = ctx->cache.emplace(key, e);
    ctx->stats.nProfileUploads++;
    ctx->stats.bytesH2D += bytes;
    ctx->stats.deviceBytes = ctx->deviceBytes;
    ctx->stats.peakDeviceBytes = std::max(ctx->stats.peakDeviceBytes, (uint64_t)ctx->deviceBytes);
    return &res.first->second;
  } catch (...) {
    /* Complete any submitted upload before releasing device storage. */
    cudaStreamSynchronize(ctx->stream);
    free_entry(ctx, e);
    throw;
  }
}

ProfileDev dev_view(const Entry *e) {
  ProfileDev d;
  d.weights = e->dWeights;
  d.codes = e->dCodes;
  d.index = e->dIndex;
  d.vectors = e->dVectors;
  d.codeDist = e->dCodeDist;
  return d;
}

void accumulate(FtGpuStats *into, const FtGpuStats &s) {
  into->nBatches += s.nBatches;
  into->nPairsGpu += s.nPairsGpu;
  into->nSitesGpu += s.nSitesGpu;
  into->nKernelLaunches += s.nKernelLaunches;
  into->bytesH2D += s.bytesH2D;
  into->bytesD2H += s.bytesD2H;
  into->tPack += s.tPack;
  into->tH2D += s.tH2D;
  into->tKernel += s.tKernel;
  into->tD2H += s.tD2H;
  into->tWait += s.tWait;
  into->tTotal += s.tTotal;
  into->nPairsGpuPair += s.nPairsGpuPair;
  into->nPairsGpuDist += s.nPairsGpuDist;
  into->nSitesPair += s.nSitesPair;
  into->nSitesDist += s.nSitesDist;
  into->nDistSeqJobs += s.nDistSeqJobs;
}

bool valid_view(FtGpuContext *ctx, const FtGpuProfileView &v, const char *op) {
  if (v.owner == NULL || v.serial == 0 || v.weights == NULL || v.codes == NULL) {
    set_error(ctx, op, "profile view has NULL weights/codes");
    return false;
  }
  if (v.nVectors < 0) {
    set_error(ctx, op, "profile view has negative nVectors");
    return false;
  }
  return true;
}

}  // namespace

/* ------------------------------------------------------------------- API */

extern "C" int ft_gpu_is_available(void) {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return count > 0 ? 1 : 0;
}

extern "C" FtGpuStatus ft_gpu_create(const FtGpuConfig *config, FtGpuContext **out_ctx) {
  if (out_ctx == NULL) return FT_GPU_ERR_INVALID;
  *out_ctx = NULL;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count <= 0) {
    cudaGetLastError();
    return FT_GPU_ERR_NO_DEVICE;
  }
  int device = (config != NULL) ? config->device_id : -1;
  if (device < 0) device = 0;
  if (device >= count) return FT_GPU_ERR_INVALID;

  FtGpuContext *ctx = NULL;
  try {
    ctx = new FtGpuContext();
  } catch (...) {
    return FT_GPU_ERR_MEMORY;
  }
  try {
  ctx->deviceId = device;
  ctx->stream = NULL;
  ctx->memoryLimit = (config != NULL) ? config->memory_limit_bytes : 0;
  ctx->deviceBytes = 0;
  ctx->idleBytes = 0;
  ctx->pinnedHostBytes = 0;
  ctx->clock = 0;
  ctx->verify = (config != NULL) ? config->verify : 0;
  ctx->statsLevel = (config != NULL) ? config->stats_level : 1;
  ctx->fatal = false;
  ctx->debugSequential = (config != NULL) ? config->debug_sequential : 0;
  ctx->haveEvents = 0;
  std::memset(ctx->ev, 0, sizeof(ctx->ev));
  std::memset(&ctx->stats, 0, sizeof(ctx->stats));
  if (cudaSetDevice(device) != cudaSuccess || cudaStreamCreate(&ctx->stream) != cudaSuccess) {
    ft_gpu_destroy(ctx);
    return FT_GPU_ERR_CUDA;
  }
  for (int i = 0; i < 4; ++i) {
    if (cudaEventCreate(&ctx->ev[i]) != cudaSuccess) {
      ft_gpu_destroy(ctx);
      return FT_GPU_ERR_CUDA;
    }
  }
  ctx->haveEvents = 1;
  cudaDeviceProp prop;
  if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "%s (sm_%d%d, %d SMs, %.0f MiB, CUDA %d.%d)",
                  prop.name, prop.major, prop.minor, prop.multiProcessorCount,
                  (double)prop.totalGlobalMem / (1024.0 * 1024.0), CUDART_VERSION / 1000,
                  (CUDART_VERSION % 1000) / 10);
    ctx->deviceInfo = buf;
  }
  *out_ctx = ctx;
  return FT_GPU_OK;
  } catch (const std::bad_alloc &) {
    ft_gpu_destroy(ctx);
    return FT_GPU_ERR_MEMORY;
  } catch (...) {
    ft_gpu_destroy(ctx);
    return FT_GPU_ERR_INTERNAL;
  }
}

extern "C" void ft_gpu_destroy(FtGpuContext *ctx) {
  if (ctx == NULL) return;
  cudaSetDevice(ctx->deviceId);
  for (auto it = ctx->cache.begin(); it != ctx->cache.end(); ++it) free_entry(ctx, it->second);
  ctx->cache.clear();
  for (auto &block : ctx->idleProfiles) cudaFree(block.second);
  ctx->idleProfiles.clear();
  for (auto &block : ctx->uploadBlocks) cudaFreeHost(block.ptr);
  ctx->scratch.release(ctx);
  for (int i = 0; i < 4; i++) if (ctx->ev[i]) cudaEventDestroy(ctx->ev[i]);
  if (ctx->stream) cudaStreamDestroy(ctx->stream);
  delete ctx;
}

extern "C" FtGpuStatus ft_gpu_device_info(FtGpuContext *ctx, char *buf, size_t buflen) {
  if (ctx == NULL || buf == NULL || buflen == 0) return FT_GPU_ERR_INVALID;
  std::snprintf(buf, buflen, "%s", ctx->deviceInfo.c_str());
  return FT_GPU_OK;
}

extern "C" FtGpuStatus ft_gpu_profile_invalidate(FtGpuContext *ctx, const void *owner,
                                                 uint64_t serial) {
  if (ctx == NULL) return FT_GPU_ERR_INVALID;
  if (cudaSetDevice(ctx->deviceId) != cudaSuccess) return FT_GPU_ERR_CUDA;
  Key key;
  key.owner = owner;
  key.serial = serial;
  auto it = ctx->cache.find(key);
  if (it != ctx->cache.end()) {
    Entry &entry = it->second;
    const size_t idleLimit = 64u * 1024u * 1024u;
    bool retained = false;
    if (ctx->memoryLimit == 0 && entry.bytes <= idleLimit - ctx->idleBytes) {
      try {
        ctx->idleProfiles.emplace(entry.bytes, entry.dWeights);
        ctx->idleBytes += entry.bytes;
        retained = true;
      } catch (...) { /* Pooling is optional; invalidation must still succeed. */ }
    }
    if (!retained) free_entry(ctx, entry);
    ctx->cache.erase(it);
    ctx->stats.nProfileInvalidations++;
  }
  ctx->stats.deviceBytes = ctx->deviceBytes;
  return FT_GPU_OK;
}

/* Drains a stream before a local staging array is destroyed on early exit. */
struct HostTransferGuard {
  FtGpuContext *ctx;
  bool active = true;
  explicit HostTransferGuard(FtGpuContext *c) : ctx(c) {}
  ~HostTransferGuard() { if (active && ctx->stream) cudaStreamSynchronize(ctx->stream); }
};

static FtGpuStatus pair_loglk_batch_impl(FtGpuContext *ctx, const FtGpuPairBatch *b, const double *transitions = NULL) {
  if (ctx == NULL || b == NULL) return FT_GPU_ERR_INVALID;
  if (ctx->fatal) return FT_GPU_ERR_CUDA;
  cuda_checked(ctx, cudaSetDevice(ctx->deviceId), "select batch device");
  if (b->nPairs < 0 || b->nPos < 0) return FT_GPU_ERR_INVALID;
  if (b->nPairs == 0) return FT_GPU_OK;
  if (b->pairs == NULL || b->out_loglk == NULL || b->nPos == 0) return FT_GPU_ERR_INVALID;
  if (b->model.nCodes != 4 && b->model.nCodes != 20) return FT_GPU_ERR_UNSUPPORTED;
  if (b->model.useMatrix &&
      (b->model.eigenval == NULL || b->model.codeFreq == NULL))
    return FT_GPU_ERR_INVALID;
  if (b->rates.rates == NULL || b->rates.ratecat == NULL) return FT_GPU_ERR_INVALID;
  if (b->rates.nRateCategories < 1 ||
      b->rates.nRateCategories > FT_GPU_MAX_RATE_CATEGORIES)
    return FT_GPU_ERR_UNSUPPORTED;

  if (!b->model.useMatrix && b->model.nCodes != 4) return FT_GPU_ERR_UNSUPPORTED;
  for (int r = 0; r < b->rates.nRateCategories; ++r)
    if (!std::isfinite(b->rates.rates[r]) || b->rates.rates[r] <= 0) return FT_GPU_ERR_INVALID;
  for (int i = 0; i < b->nPos; ++i)
    if (b->rates.ratecat[i] >= (unsigned)b->rates.nRateCategories) return FT_GPU_ERR_INVALID;
  for (int i = 0; i < b->nPairs; ++i)
    if (!std::isfinite(b->pairs[i].length)) return FT_GPU_ERR_INVALID;

  const int64_t tStart = now_us();
  const int nCodes = b->model.nCodes;
  const int nPos = b->nPos;
  const int nPairs = b->nPairs;
  FtGpuStats local;
  std::memset(&local, 0, sizeof(local));

  /* overflow checks */
  const size_t nSiteTotal = (size_t)nPairs * (size_t)nPos;
  if (nPos > 0 && nSiteTotal / (size_t)nPos != (size_t)nPairs) return FT_GPU_ERR_OVERFLOW;

  if (nSiteTotal > SIZE_MAX / sizeof(double)) return FT_GPU_ERR_OVERFLOW;
  size_t nTransition=(size_t)nPairs*b->rates.nRateCategories*(b->model.useMatrix?nCodes:2);
  if(transitions)for(size_t i=0;i<nTransition;i++)
    if(!std::isfinite(transitions[i])||transitions[i]<0)return FT_GPU_ERR_INVALID;
  bool wantSite = b->out_site != NULL || transitions != NULL;
  for (int i = 0; i < nPairs; i++)
    if (b->pairs[i].site_likelihoods != NULL && !wantSite) return FT_GPU_ERR_INVALID;

  if (ctx->haveEvents) cudaEventRecord(ctx->ev[0], ctx->stream);
  const int64_t tPack0 = now_us();
  std::vector<DevPair> devPairs((size_t)nPairs);
  HostTransferGuard inputGuard(ctx);
  ctx->pinned.clear();
  for (int i = 0; i < nPairs; i++) {
    const FtGpuPair &p = b->pairs[i];
    if (!valid_view(ctx, p.A, "ft_gpu_pair_loglk_batch") ||
        !valid_view(ctx, p.B, "ft_gpu_pair_loglk_batch"))
      return FT_GPU_ERR_INVALID;
    const Entry *ea = get_profile(ctx, p.A, nPos, nCodes, "ft_gpu_pair_loglk_batch");
    if (ea == NULL) return FT_GPU_ERR_MEMORY;
    ctx->pinned.push_back(ea->key);
    const Entry *eb = get_profile(ctx, p.B, nPos, nCodes, "ft_gpu_pair_loglk_batch");
    if (eb == NULL) return FT_GPU_ERR_MEMORY;
    ctx->pinned.push_back(eb->key);
    devPairs[i].A = dev_view(ea);
    devPairs[i].B = dev_view(eb);
    devPairs[i].length = p.length;
    devPairs[i].wantSite = wantSite ? 1 : 0;
  }
  const int64_t tPack1 = now_us();

  /* device buffers */
  Scratch &sc = ctx->scratch;
  const char *kOp = "ft_gpu_pair_loglk_batch";
  bool ok = sc.dist.ensure(ctx, (size_t)nPairs, kOp);
  ok = ok && sc.pairs.ensure(ctx, (size_t)nPairs, kOp);
  if (wantSite) ok = ok && sc.site.ensure(ctx, nSiteTotal, kOp);
  ok = ok && sc.ratecat.ensure(ctx, (size_t)nPos, kOp);
  ok = ok && sc.rates.ensure(ctx, (size_t)b->rates.nRateCategories, kOp);
  if (b->model.useMatrix) {
    ok = ok && sc.eigenval.ensure(ctx, (size_t)nCodes, kOp);
    ok = ok && sc.codeFreq.ensure(ctx, (size_t)FT_GPU_CODE_ROWS * FT_GPU_CODE_STRIDE, kOp);
  }
  if(transitions)ok=ok && sc.transitions.ensure(ctx,nTransition,kOp);
  if (!ok) return FT_GPU_ERR_MEMORY;
  DevPair *dPairs = sc.pairs.ptr;
  double *dLoglk = sc.dist.ptr;
  double *dSite = wantSite ? sc.site.ptr : NULL;
  unsigned int *dRatecatU = sc.ratecat.ptr;
  double *dRates = sc.rates.ptr;
  double *dEigenval = b->model.useMatrix ? sc.eigenval.ptr : NULL;
  double *dCodeFreq = b->model.useMatrix ? sc.codeFreq.ptr : NULL;
  local.bytesH2D += (size_t)nPairs * sizeof(DevPair);
  local.bytesH2D += (size_t)nPos * sizeof(unsigned int);
  local.bytesH2D += (size_t)b->rates.nRateCategories * sizeof(double);
  if (b->model.useMatrix) {
    local.bytesH2D += (size_t)nCodes * sizeof(double);
    local.bytesH2D += (size_t)FT_GPU_CODE_ROWS * FT_GPU_CODE_STRIDE * sizeof(double);
  }

  cuda_checked(ctx, cudaMemcpyAsync(dPairs, devPairs.data(), (size_t)nPairs * sizeof(DevPair),
                  cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
  cuda_checked(ctx, cudaMemcpyAsync(dRatecatU, b->rates.ratecat, (size_t)nPos * sizeof(unsigned int),
                  cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
  cuda_checked(ctx, cudaMemcpyAsync(dRates, b->rates.rates, (size_t)b->rates.nRateCategories * sizeof(double),
                  cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
  if (b->model.useMatrix) {
    cuda_checked(ctx, cudaMemcpyAsync(dEigenval, b->model.eigenval, (size_t)nCodes * sizeof(double),
                    cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
    cuda_checked(ctx, cudaMemcpyAsync(dCodeFreq, b->model.codeFreq,
                    (size_t)FT_GPU_CODE_ROWS * FT_GPU_CODE_STRIDE * sizeof(double),
                    cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
  }
  if(transitions) {
    cuda_checked(ctx,cudaMemcpyAsync(sc.transitions.ptr,transitions,nTransition*sizeof(double),cudaMemcpyHostToDevice,ctx->stream),"ordered transitions");
    local.bytesH2D+=nTransition*sizeof(double);
  }
  const double *dTransition=transitions?sc.transitions.ptr:NULL;
  if (ctx->haveEvents) cudaEventRecord(ctx->ev[1], ctx->stream);
  const int64_t tH2D1 = now_us();

  const int nThreads = ctx->debugSequential ? 1 : kThreads;
  size_t sharedBytes;
  if (b->model.useMatrix)
    sharedBytes = (size_t)b->rates.nRateCategories * nCodes * sizeof(double);
  else
    sharedBytes = (size_t)b->rates.nRateCategories * 2 * sizeof(double);

  if (b->model.useMatrix) {
    if (nCodes == 4)
      pair_loglk_matrix_kernel<4><<<nPairs, nThreads, sharedBytes, ctx->stream>>>(
          dPairs, dRatecatU, nPos, dRates, b->rates.nRateCategories, dEigenval, dCodeFreq,
          dLoglk, dSite, dTransition);
    else
      pair_loglk_matrix_kernel<20><<<nPairs, nThreads, sharedBytes, ctx->stream>>>(
          dPairs, dRatecatU, nPos, dRates, b->rates.nRateCategories, dEigenval, dCodeFreq,
          dLoglk, dSite, dTransition);
  } else {
    if (nCodes != 4) return FT_GPU_ERR_UNSUPPORTED;
    pair_loglk_jc_kernel<<<nPairs, nThreads, sharedBytes, ctx->stream>>>(
        dPairs, dRatecatU, nPos, dRates, b->rates.nRateCategories, dLoglk, dSite, dTransition);
  }
  local.nKernelLaunches++;
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    ctx->fatal = true;
    set_error(ctx, "ft_gpu_pair_loglk_batch", "kernel launch: " + cuda_err(launchErr));
    return FT_GPU_ERR_CUDA;
  }
  if (ctx->haveEvents) cudaEventRecord(ctx->ev[2], ctx->stream);
  const int64_t tKernel1 = now_us();

  cudaError_t err = cudaStreamSynchronize(ctx->stream);
  const int64_t tWait1 = now_us();
  if (err != cudaSuccess) {
    ctx->fatal = true;
    set_error(ctx, "ft_gpu_pair_loglk_batch", "kernel execution: " + cuda_err(err));
    return FT_GPU_ERR_CUDA;
  }
  /* No caller output is changed until every result copy and validation succeeds. */
  std::vector<double> hostLoglk((size_t)nPairs);
  std::vector<double> hostSite(wantSite ? nSiteTotal : 0);
  inputGuard.active = false;
  HostTransferGuard outputGuard(ctx);
  err = cudaMemcpyAsync(hostLoglk.data(), dLoglk, (size_t)nPairs * sizeof(double),
                        cudaMemcpyDeviceToHost, ctx->stream);
  if (err == cudaSuccess && wantSite)
    err = cudaMemcpyAsync(hostSite.data(), dSite, nSiteTotal * sizeof(double),
                          cudaMemcpyDeviceToHost, ctx->stream);
  /* The event must be recorded *before* the synchronisation, otherwise
     cudaEventElapsedTime() can be queried before the event completes. */
  if (ctx->haveEvents) cudaEventRecord(ctx->ev[3], ctx->stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(ctx->stream);
  const int64_t tD2H1 = now_us();
  if (err != cudaSuccess) {
    ctx->fatal = true;
    set_error(ctx, "ft_gpu_pair_loglk_batch", "result copy: " + cuda_err(err));
    return FT_GPU_ERR_CUDA;
  }

  outputGuard.active = false;
  for (double value : hostLoglk)
    if (!std::isfinite(value))
      fail(ctx, FT_GPU_ERR_INTERNAL, "PairLogLk", "non-finite result (invalid/non-positive site likelihood)");
  for (double value : hostSite)
    if (!(value > 0.0) || !std::isfinite(value))
      fail(ctx, FT_GPU_ERR_INTERNAL, "PairLogLk", "non-positive or non-finite site likelihood");
  if(transitions)for(int pair=0;pair<nPairs;pair++) {
    double product=1.0,correction=0.0;
    for(int site=0;site<nPos;site++) {
      const FtGpuPair &p=b->pairs[pair];
      if(b->model.useMatrix && p.A.weights[site]==0 && p.B.weights[site]==0 &&
         p.A.codes[site]==FT_GPU_NOCODE && p.B.codes[site]==FT_GPU_NOCODE)continue;
      product*=hostSite[(size_t)pair*nPos+site];
      if(!(product>0)||!std::isfinite(product))return FT_GPU_ERR_INTERNAL;
      while(product<kLkUnderflow){product*=kLkUnderflowInv;correction-=kLogLkUnderflow;}
      if(b->model.useMatrix)while(product>kLkUnderflowInv){product*=kLkUnderflow;correction+=kLogLkUnderflow;}
    }
    hostLoglk[pair]=correction+std::log(product);
    if(!std::isfinite(hostLoglk[pair]))return FT_GPU_ERR_INTERNAL;
  }
  std::memcpy(b->out_loglk, hostLoglk.data(), (size_t)nPairs * sizeof(double));
  if (b->out_site) std::memcpy(b->out_site, hostSite.data(), nSiteTotal * sizeof(double));

  local.nBatches = 1;
  local.nPairsGpu = (uint64_t)nPairs;
  local.nSitesGpu = (uint64_t)nSiteTotal;
  local.nPairsGpuPair = (uint64_t)nPairs;
  local.nSitesPair = (uint64_t)nSiteTotal;
  local.bytesD2H += (size_t)nPairs * sizeof(double);
  if (wantSite) local.bytesD2H += nSiteTotal * sizeof(double);
  local.tPack = (tPack1 - tPack0) / 1e6;
  local.tH2D = (tH2D1 - tPack1) / 1e6;
  local.tKernel = (tKernel1 - tH2D1) / 1e6;
  local.tWait = (tWait1 - tKernel1) / 1e6;
  local.tD2H = (tD2H1 - tWait1) / 1e6;
  local.tTotal = (tD2H1 - tStart) / 1e6;
  if (ctx->haveEvents) {
    float ms01 = 0, ms12 = 0, ms23 = 0;
    cudaEventElapsedTime(&ms01, ctx->ev[0], ctx->ev[1]);
    cudaEventElapsedTime(&ms12, ctx->ev[1], ctx->ev[2]);
    cudaEventElapsedTime(&ms23, ctx->ev[2], ctx->ev[3]);
    local.tH2D = ms01 / 1e3;
    local.tKernel = ms12 / 1e3;
    local.tD2H = ms23 / 1e3;
    local.tWait = local.tTotal - (local.tPack + local.tH2D + local.tKernel + local.tD2H);
    if (local.tWait < 0) local.tWait = 0;
  }
  ctx->stats.tKernelPair += local.tKernel;
  if (trace_enabled())
    fprintf(stderr,
            "# FT_GPU_TRACE pairloglk nPairs=%d nPos=%d codes=%d matrix=%d matrixmodel=%d "
            "kernel=%.3fms h2d=%.3fms total=%.3fms\n",
            nPairs, nPos, nCodes, b->model.useMatrix, b->model.useMatrix, local.tKernel * 1e3,
            local.tH2D * 1e3, local.tTotal * 1e3);
  accumulate(&ctx->stats, local);
  ctx->stats.deviceBytes = ctx->deviceBytes;
  return FT_GPU_OK;
}

static FtGpuStatus dist_batch_impl(FtGpuContext *ctx, const FtGpuDistBatch *b, bool ordered = false) {
  if (ctx == NULL || b == NULL) return FT_GPU_ERR_INVALID;
  if (ctx->fatal) return FT_GPU_ERR_CUDA;
  cuda_checked(ctx, cudaSetDevice(ctx->deviceId), "select batch device");
  if (b->nPairs < 0 || b->nPos < 0) return FT_GPU_ERR_INVALID;
  if (b->nPairs == 0) return FT_GPU_OK;
  if (b->pairs == NULL || b->out_dist == NULL || b->out_weight == NULL || b->nPos == 0)
    return FT_GPU_ERR_INVALID;
  if (b->nCodes != 4 && b->nCodes != 20) return FT_GPU_ERR_UNSUPPORTED;
  if (b->useMatrix && (b->dmat_distances == NULL || b->dmat_codeFreq == NULL ||
                       b->dmat_eigenval == NULL)) return FT_GPU_ERR_INVALID;

  const int64_t tStart = now_us();
  const int nPos = b->nPos;
  const int nPairs = b->nPairs;
  FtGpuStats local;
  std::memset(&local, 0, sizeof(local));

  if (ctx->haveEvents) cudaEventRecord(ctx->ev[0], ctx->stream);
  const int64_t tPack0 = now_us();
  std::vector<DevDistPair> devPairs((size_t)nPairs);
  HostTransferGuard inputGuard(ctx);
  /* Sequence (code/code) jobs need their codes on the device: a kernel must
     never dereference a host pointer directly.  All of them are packed into a
     single contiguous upload. */
  size_t seqBytes = 0;
  for (int i = 0; i < nPairs; i++) {
    const FtGpuDistPair &p = b->pairs[i];
    if (!p.isSeq) continue;
    int rawA = (p.codesA != NULL);
    int rawB = (p.codesB != NULL);
    if (rawA != rawB) return FT_GPU_ERR_INVALID;
    if (rawA) {
      if (!valid_codes(p.codesA, nPos, b->nCodes) || !valid_codes(p.codesB, nPos, b->nCodes))
        return FT_GPU_ERR_INVALID;
      if (seqBytes > SIZE_MAX - 2 * (size_t)nPos) return FT_GPU_ERR_OVERFLOW;
      seqBytes += 2 * (size_t)nPos;
    } else if (p.A.owner == NULL || p.B.owner == NULL || p.A.codes == NULL ||
               p.B.codes == NULL) {
      return FT_GPU_ERR_INVALID;
    }
  }
  ctx->pinned.clear();
  bool seqPackOk = true;
  if (seqBytes > 0) {
    seqPackOk = ctx->scratch.seqCodes.ensure(ctx, seqBytes, "ft_gpu_dist_batch");
    if (seqPackOk) {
      try {
        ctx->scratch.hostSeqCodes.resize(seqBytes);
      } catch (...) {
        set_error(ctx, "ft_gpu_dist_batch", "host staging allocation failed");
        return FT_GPU_ERR_MEMORY;
      }
    } else {
      return FT_GPU_ERR_MEMORY;
    }
  }
  size_t seqOffset = 0;
  ctx->pinned.clear();
  for (int i = 0; i < nPairs; i++) {
    const FtGpuDistPair &p = b->pairs[i];
    std::memset(&devPairs[i], 0, sizeof(DevDistPair));
    devPairs[i].isSeq = p.isSeq;
    if (p.isSeq && p.codesA != NULL) {
      unsigned char *dst = ctx->scratch.hostSeqCodes.data() + seqOffset;
      std::memcpy(dst, p.codesA, (size_t)nPos);
      std::memcpy(dst + nPos, p.codesB, (size_t)nPos);
      devPairs[i].codesA = ctx->scratch.seqCodes.ptr + seqOffset;
      devPairs[i].codesB = ctx->scratch.seqCodes.ptr + seqOffset + nPos;
      seqOffset += 2 * (size_t)nPos;
    } else if (p.isSeq) {
      if (!valid_view(ctx, p.A, "ft_gpu_dist_batch") ||
          !valid_view(ctx, p.B, "ft_gpu_dist_batch") || p.A.nVectors != 0 || p.B.nVectors != 0)
        return FT_GPU_ERR_INVALID;
      /* code-only profiles: keep them in the device cache */
      const Entry *ea = get_profile(ctx, p.A, nPos, b->nCodes, "ft_gpu_dist_batch");
      if (ea == NULL) return FT_GPU_ERR_MEMORY;
      ctx->pinned.push_back(ea->key);
      const Entry *eb = get_profile(ctx, p.B, nPos, b->nCodes, "ft_gpu_dist_batch");
      if (eb == NULL) return FT_GPU_ERR_MEMORY;
      ctx->pinned.push_back(eb->key);
      devPairs[i].A = dev_view(ea);
      devPairs[i].B = dev_view(eb);
    } else {
      if (!valid_view(ctx, p.A, "ft_gpu_dist_batch") ||
          !valid_view(ctx, p.B, "ft_gpu_dist_batch"))
        return FT_GPU_ERR_INVALID;
      const Entry *ea = get_profile(ctx, p.A, nPos, b->nCodes, "ft_gpu_dist_batch");
      if (ea == NULL) return FT_GPU_ERR_MEMORY;
      ctx->pinned.push_back(ea->key);
      const Entry *eb = get_profile(ctx, p.B, nPos, b->nCodes, "ft_gpu_dist_batch");
      if (eb == NULL) return FT_GPU_ERR_MEMORY;
      ctx->pinned.push_back(eb->key);
      devPairs[i].A = dev_view(ea);
      devPairs[i].B = dev_view(eb);
    }
  }
  const int64_t tPack1 = now_us();

  Scratch &sc = ctx->scratch;
  const char *kOp = "ft_gpu_dist_batch";
  bool ok = sc.dist.ensure(ctx, (size_t)nPairs, kOp);
  ok = ok && sc.weight.ensure(ctx, (size_t)nPairs, kOp);
  ok = ok && sc.dpairs.ensure(ctx, (size_t)nPairs, kOp);
  if (b->useMatrix) {
    ok = ok && sc.dmatDist.ensure(ctx, (size_t)FT_GPU_MAXCODES * FT_GPU_MAXCODES, kOp);
    ok = ok && sc.dmatCodeFreq.ensure(ctx, (size_t)FT_GPU_MAXCODES * FT_GPU_MAXCODES, kOp);
    if (b->dmat_eigenval != NULL) ok = ok && sc.dmatEig.ensure(ctx, (size_t)b->nCodes, kOp);
  }
  if (!ok) return FT_GPU_ERR_MEMORY;
  DevDistPair *dPairs = sc.dpairs.ptr;
  double *dDist = sc.dist.ptr;
  double *dWeight = sc.weight.ptr;
  double *dDmatDist = b->useMatrix ? sc.dmatDist.ptr : NULL;
  double *dDmatCodeFreq = b->useMatrix ? sc.dmatCodeFreq.ptr : NULL;
  double *dDmatEig = (b->useMatrix && b->dmat_eigenval != NULL) ? sc.dmatEig.ptr : NULL;
  local.bytesH2D += (size_t)nPairs * sizeof(DevDistPair);
  cuda_checked(ctx, cudaMemcpyAsync(dPairs, devPairs.data(), (size_t)nPairs * sizeof(DevDistPair),
                  cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
  if (seqBytes > 0) {
    cuda_checked(ctx, cudaMemcpyAsync(ctx->scratch.seqCodes.ptr, ctx->scratch.hostSeqCodes.data(), seqBytes,
                    cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
    local.bytesH2D += seqBytes;
  }
  if (b->useMatrix) {
    cuda_checked(ctx, cudaMemcpyAsync(dDmatDist, b->dmat_distances,
                    (size_t)FT_GPU_MAXCODES * FT_GPU_MAXCODES * sizeof(double),
                    cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
    cuda_checked(ctx, cudaMemcpyAsync(dDmatCodeFreq, b->dmat_codeFreq,
                    (size_t)FT_GPU_MAXCODES * FT_GPU_MAXCODES * sizeof(double),
                    cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
    if (b->dmat_eigenval != NULL)
      cuda_checked(ctx, cudaMemcpyAsync(dDmatEig, b->dmat_eigenval, (size_t)b->nCodes * sizeof(double),
                      cudaMemcpyHostToDevice, ctx->stream), "batch H2D copy");
    local.bytesH2D += (size_t)FT_GPU_MAXCODES * FT_GPU_MAXCODES * sizeof(double);
    local.bytesH2D += (size_t)FT_GPU_MAXCODES * FT_GPU_MAXCODES * sizeof(double);
    local.bytesH2D += (size_t)b->nCodes * sizeof(double);
  }
  if (ctx->haveEvents) cudaEventRecord(ctx->ev[1], ctx->stream);
  const int64_t tH2D1 = now_us();

  if(ordered) {
    if(nPos>65535*128 || (size_t)nPairs>SIZE_MAX/(size_t)nPos/2/sizeof(double))return FT_GPU_ERR_UNSUPPORTED;
    for(int i=0;i<nPairs;i++)if(b->pairs[i].isSeq)return FT_GPU_ERR_UNSUPPORTED;
    if(!sc.site.ensure(ctx,(size_t)nPairs*nPos*2,"ordered distance"))return FT_GPU_ERR_MEMORY;
    dim3 grid(nPairs,(nPos+127)/128);
    dist_ordered_sites<<<grid,128,0,ctx->stream>>>(dPairs,nPos,b->nCodes,b->useMatrix,dDmatDist,dDmatEig,dDmatCodeFreq,sc.site.ptr);
    cuda_checked(ctx,cudaGetLastError(),"ordered distance sites");local.nKernelLaunches++;
    dist_ordered_reduce<<<nPairs,1,0,ctx->stream>>>(dPairs,nPos,sc.site.ptr,dDist,dWeight);
  } else if (ctx->debugSequential) {
    dist_kernel<<<nPairs, 1, 0, ctx->stream>>>(dPairs, nPos, b->nCodes, b->useMatrix, dDmatDist,
                                               dDmatEig, dDmatCodeFreq, dDist, dWeight);
  } else if (nPos <= kWarpKernelMaxSites) {
    const int warpsPerBlock = 8;
    const int blocks = (nPairs - 1) / warpsPerBlock + 1;
    dist_warp_kernel<<<blocks, warpsPerBlock * 32, 0, ctx->stream>>>(
        dPairs, nPairs, nPos, b->nCodes, b->useMatrix, dDmatDist, dDmatEig, dDmatCodeFreq,
        dDist, dWeight);
  } else {
    dist_kernel<<<nPairs, kDistThreads, 0, ctx->stream>>>(
        dPairs, nPos, b->nCodes, b->useMatrix, dDmatDist, dDmatEig, dDmatCodeFreq, dDist,
        dWeight);
  }
  local.nKernelLaunches++;
  cudaError_t launchErr = cudaGetLastError();
  if (launchErr != cudaSuccess) {
    ctx->fatal = true;
    set_error(ctx, "ft_gpu_dist_batch", "kernel launch: " + cuda_err(launchErr));
    return FT_GPU_ERR_CUDA;
  }
  if (ctx->haveEvents) cudaEventRecord(ctx->ev[2], ctx->stream);
  const int64_t tKernel1 = now_us();

  cudaError_t err = cudaStreamSynchronize(ctx->stream);
  const int64_t tWait1 = now_us();
  if (err != cudaSuccess) {
    ctx->fatal = true;
    set_error(ctx, "ft_gpu_dist_batch", "kernel execution: " + cuda_err(err));
    return FT_GPU_ERR_CUDA;
  }
  std::vector<double> hostDist((size_t)nPairs), hostWeight((size_t)nPairs);
  inputGuard.active = false;
  HostTransferGuard outputGuard(ctx);
  err = cudaMemcpyAsync(hostDist.data(), dDist, (size_t)nPairs * sizeof(double),
                        cudaMemcpyDeviceToHost, ctx->stream);
  if (err == cudaSuccess)
    err = cudaMemcpyAsync(hostWeight.data(), dWeight, (size_t)nPairs * sizeof(double),
                          cudaMemcpyDeviceToHost, ctx->stream);
  if (ctx->haveEvents) cudaEventRecord(ctx->ev[3], ctx->stream);
  if (err == cudaSuccess) err = cudaStreamSynchronize(ctx->stream);
  const int64_t tD2H1 = now_us();
  if (err != cudaSuccess) {
    ctx->fatal = true;
    set_error(ctx, "ft_gpu_dist_batch", "result copy: " + cuda_err(err));
    return FT_GPU_ERR_CUDA;
  }

  outputGuard.active = false;
  for (int i = 0; i < nPairs; ++i)
    if (!std::isfinite(hostDist[i]) || !std::isfinite(hostWeight[i]))
      fail(ctx, FT_GPU_ERR_INTERNAL, "ProfileDist", "non-finite distance result");
  std::memcpy(b->out_dist, hostDist.data(), (size_t)nPairs * sizeof(double));
  std::memcpy(b->out_weight, hostWeight.data(), (size_t)nPairs * sizeof(double));

  local.nBatches = 1;
  local.nPairsGpu = (uint64_t)nPairs;
  local.nSitesGpu = (uint64_t)nPairs * (uint64_t)nPos;
  local.nPairsGpuDist = (uint64_t)nPairs;
  local.nSitesDist = (uint64_t)nPairs * (uint64_t)nPos;
  local.nDistSeqJobs = 0;
  for (int t = 0; t < nPairs; t++)
    if (b->pairs[t].isSeq) local.nDistSeqJobs++;
  local.bytesD2H += 2 * (size_t)nPairs * sizeof(double);
  local.tPack = (tPack1 - tPack0) / 1e6;
  local.tH2D = (tH2D1 - tPack1) / 1e6;
  local.tKernel = (tKernel1 - tH2D1) / 1e6;
  local.tWait = (tWait1 - tKernel1) / 1e6;
  local.tD2H = (tD2H1 - tWait1) / 1e6;
  local.tTotal = (tD2H1 - tStart) / 1e6;
  if (ctx->haveEvents) {
    float ms01 = 0, ms12 = 0, ms23 = 0;
    cudaEventElapsedTime(&ms01, ctx->ev[0], ctx->ev[1]);
    cudaEventElapsedTime(&ms12, ctx->ev[1], ctx->ev[2]);
    cudaEventElapsedTime(&ms23, ctx->ev[2], ctx->ev[3]);
    local.tH2D = ms01 / 1e3;
    local.tKernel = ms12 / 1e3;
    local.tD2H = ms23 / 1e3;
    local.tWait = local.tTotal - (local.tPack + local.tH2D + local.tKernel + local.tD2H);
    if (local.tWait < 0) local.tWait = 0;
  }
  ctx->stats.tKernelDist += local.tKernel;
  if (trace_enabled())
    fprintf(stderr,
            "# FT_GPU_TRACE dist nPairs=%d nPos=%d seqJobs=%llu matrix=%d kernel=%.3fms "
            "h2d=%.3fms total=%.3fms\n",
            nPairs, nPos, (unsigned long long)local.nDistSeqJobs, b->useMatrix,
            local.tKernel * 1e3, local.tH2D * 1e3, local.tTotal * 1e3);
  accumulate(&ctx->stats, local);
  ctx->stats.deviceBytes = ctx->deviceBytes;
  return FT_GPU_OK;
}

static FtGpuStatus finish_failed_call(FtGpuContext *ctx, FtGpuStatus status, const char *op) {
  if (ctx == NULL) return status;
  if (ctx->stream && !ctx->fatal) {
    cudaError_t err = cudaStreamSynchronize(ctx->stream);
    if (err != cudaSuccess) {
      ctx->fatal = true;
      set_error(ctx, op, cuda_err(err));
      status = FT_GPU_ERR_CUDA;
    }
  }
  ctx->pinned.clear();
  reset_uploads(ctx);
  if (ctx->lastError.empty()) set_error(ctx, op, ft_gpu_status_string(status));
  return status;
}

#define FT_BATCH_WRAPPER(name, type, impl) \
extern "C" FtGpuStatus name(FtGpuContext *ctx, const type *b) { \
  if (ctx) ctx->lastError.clear(); \
  FtGpuStatus status; \
  try { status = impl(ctx, b); } \
  catch (const BackendFailure &e) { status = e.status; } \
  catch (const std::bad_alloc &) { status = FT_GPU_ERR_MEMORY; } \
  catch (const std::length_error &) { status = FT_GPU_ERR_OVERFLOW; } \
  catch (...) { status = FT_GPU_ERR_INTERNAL; } \
  if (status != FT_GPU_OK) { \
    try { return finish_failed_call(ctx, status, #name); } \
    catch (...) { if (ctx) { ctx->pinned.clear(); reset_uploads(ctx); } return status; } \
  } \
  if (ctx) { ctx->pinned.clear(); reset_uploads(ctx); } \
  return FT_GPU_OK; \
}
static FtGpuStatus ordered_pair_impl(FtGpuContext *ctx,const FtGpuOrderedPairBatch *b) {
  if(!b || !b->transitions)return FT_GPU_ERR_INVALID;
  return pair_loglk_batch_impl(ctx,&b->batch,b->transitions);
}
FT_BATCH_WRAPPER(ft_gpu_pair_loglk_ordered, FtGpuOrderedPairBatch, ordered_pair_impl)
FT_BATCH_WRAPPER(ft_gpu_pair_loglk_batch, FtGpuPairBatch, pair_loglk_batch_impl)
static FtGpuStatus ordered_dist_impl(FtGpuContext *ctx,const FtGpuDistBatch *b) {
  return dist_batch_impl(ctx,b,true);
}
FT_BATCH_WRAPPER(ft_gpu_dist_ordered, FtGpuDistBatch, ordered_dist_impl)
FT_BATCH_WRAPPER(ft_gpu_dist_batch, FtGpuDistBatch, dist_batch_impl)
#undef FT_BATCH_WRAPPER

extern "C" void ft_gpu_stats_get(FtGpuContext *ctx, FtGpuStats *out) {
  if (ctx == NULL || out == NULL) return;
  *out = ctx->stats;
  out->deviceBytes = ctx->deviceBytes;
  out->peakDeviceBytes = ctx->stats.peakDeviceBytes;
}

extern "C" void ft_gpu_stats_reset(FtGpuContext *ctx) {
  if (ctx == NULL) return;
  std::memset(&ctx->stats, 0, sizeof(ctx->stats));
  ctx->stats.deviceBytes = ctx->deviceBytes;
  ctx->stats.peakDeviceBytes = ctx->deviceBytes;
}

extern "C" const char *ft_gpu_last_error(FtGpuContext *ctx) {
  if (ctx == NULL) return "no context";
  return ctx->lastError.c_str();
}

extern "C" const char *ft_gpu_status_string(FtGpuStatus status) {
  switch (status) {
    case FT_GPU_OK: return "ok";
    case FT_GPU_ERR_INVALID: return "invalid argument";
    case FT_GPU_ERR_NO_DEVICE: return "no CUDA device / backend not built";
    case FT_GPU_ERR_UNSUPPORTED: return "unsupported configuration";
    case FT_GPU_ERR_MEMORY: return "out of memory";
    case FT_GPU_ERR_CUDA: return "CUDA error";
    case FT_GPU_ERR_OVERFLOW: return "size overflow";
    case FT_GPU_ERR_VERIFY: return "verification mismatch";
    case FT_GPU_ERR_INTERNAL: return "internal error";
  }
  return "unknown";
}

extern "C" void ft_gpu_set_stats_level(FtGpuContext *ctx, int level) {
  if (ctx != NULL) ctx->statsLevel = level;
}

#include "ft_gpu_resident.cuh"

#include "ft_gpu_jc.cuh"
