/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ft_gpu.h -- C ABI for the optional FastTree CUDA backend.
 *
 * This header is plain C (C89-compatible declarations, no STL, no CUDA types)
 * so that the original FastTree.c translation unit can include it without
 * becoming C++ and without requiring a CUDA compiler.
 *
 * Contract summary (see docs/IMPLEMENTATION.md for the full text):
 *  - All input host arrays are read only for the duration of the call: the backend copies what it needs into
 *    its own device/context storage before returning.
 *  - Calls are synchronous: when a batch function returns FT_GPU_OK the
 *    outputs in the result structs are valid.
 *  - A batch of 0 pairs is legal and does nothing.
 *  - On any error the function returns a status != FT_GPU_OK and NO output is
 *    committed: callers must keep their previous state and may fall back to
 *    the CPU implementation.
 *  - The backend never invokes C++ exceptions across this ABI.
 */
#ifndef FT_GPU_H
#define FT_GPU_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FT_GPU_NOCODE 127
#define FT_GPU_MAXCODES 20
/* codeFreq/gap rows are indexed by a raw code value, and NOCODE (127) is a
   valid index, so the tables have 128 rows.  Rows are MAXCODES wide, matching
   FastTree's `numeric_t codeFreq[NOCODE+1][MAXCODES]` and
   `distance_matrix_t`, so host arrays can be passed without repacking. */
#define FT_GPU_CODE_ROWS 128
#define FT_GPU_CODE_STRIDE FT_GPU_MAXCODES
#define FT_GPU_MAX_RATE_CATEGORIES 64
#define FT_GPU_ERROR_MESSAGE_SIZE 512

typedef enum {
  FT_GPU_OK = 0,
  FT_GPU_ERR_INVALID = 1,      /* bad argument (NULL, negative size, unknown code) */
  FT_GPU_ERR_NO_DEVICE = 2,    /* no usable CUDA device / backend not built */
  FT_GPU_ERR_UNSUPPORTED = 3,  /* well-formed but not implemented (e.g. nCodes!=4/20) */
  FT_GPU_ERR_MEMORY = 4,       /* host or device allocation failed / budget exceeded */
  FT_GPU_ERR_CUDA = 5,         /* unrecoverable CUDA error; context must be destroyed */
  FT_GPU_ERR_OVERFLOW = 6,     /* size computation would overflow size_t */
  FT_GPU_ERR_VERIFY = 7,       /* --gpu-verify comparison failed */
  FT_GPU_ERR_INTERNAL = 8
} FtGpuStatus;

typedef struct FtGpuContext FtGpuContext;

typedef struct {
  int device_id;              /* -1 = default */
  size_t memory_limit_bytes;  /* cache + scratch limit; 0 = no explicit limit */
  int verify;                 /* 1 = recompute the batch on the CPU and compare */
  int stats_level;            /* 0 = quiet, 1 = counters, 2 = counters+timers */
  /* Diagnostic only: run one thread per pair so that each pair's product is
     accumulated in the same order as the CPU loop.  This isolates the effect of
     the parallel reduction from the effect of exp()/FMA differences.  It is
     much slower and must not be used for performance measurements. */
  int debug_sequential;
} FtGpuConfig;

/* A read-only view of one host profile.
 *
 * "owner" plus "serial" identify the *object contents*: the backend caches
 * device copies under that pair.  A host profile whose arrays are rewritten in
 * place must be given a new serial (or invalidated) before it is used again.
 * Address reuse after free is therefore harmless: the new object gets a new
 * serial, and ft_gpu_profile_invalidate() also drops the stale entry.
 */
typedef struct {
  const void *owner;          /* identity of the host object (e.g. profile_t*) */
  uint64_t serial;            /* changes whenever the contents change */
  const double *weights;      /* [nPos] */
  const unsigned char *codes; /* [nPos] */
  const double *vectors;      /* [nVectors*nCodes] or NULL */
  int nVectors;
  const double *codeDist;     /* [nPos*nCodes] optional (out-profiles only) */
} FtGpuProfileView;

typedef struct {
  int nCodes;                 /* 4 (nucleotide) or 20 (protein) */
  int useMatrix;              /* 0 = Jukes-Cantor (transmat == NULL) */
  const double *eigenval;     /* [nCodes], matrix models only */
  const double *codeFreq;     /* [FT_GPU_CODE_ROWS*FT_GPU_CODE_STRIDE], row-major
                                 (transmat->codeFreq[0] verbatim) */
} FtGpuModel;

typedef struct {
  int nRateCategories;        /* >= 1 */
  const double *rates;        /* [nRateCategories] */
  const unsigned int *ratecat;/* [nPos] */
} FtGpuRates;

/* Experimental JC posterior: table [4*nRates] is CPU PSameA, PDiffA,
   PSameB, PDiffB. Outputs are dense [nPos] weights/codes and [nPos*4]
   vectors (zero for simple sites). Caller compacts vectors in site order.
   Synchronous, transactional, context memory budget applies. */
FtGpuStatus ft_gpu_jc_posterior(FtGpuContext *ctx, int nPos, int nRates,
 const FtGpuProfileView *a, const FtGpuProfileView *b,
 const unsigned int *ratecat, const double *table,
 double *weights, unsigned char *codes, double *vectors);

/* ---- PairLogLk batch ---- */

typedef struct {
  FtGpuProfileView A;
  FtGpuProfileView B;
  double length;              /* branch length passed to PairLogLk */
  double *site_likelihoods;   /* optional marker: requires out_site; never modified
                                by this API. Caller multiplies the contributions. */
} FtGpuPair;

typedef struct {
  int nPos;                  /* > 0 for a nonempty batch */
  FtGpuModel model;
  FtGpuRates rates;
  int nPairs;
  const FtGpuPair *pairs;
  /* outputs */
  double *out_loglk;          /* [nPairs] log-likelihood contribution */
  double *out_site;           /* optional [nPairs*nPos] per-site lkAB, or NULL;
                                requesting it is sufficient, even without markers */
} FtGpuPairBatch;
/* Search experiment: CPU-generated transitions, GPU site arithmetic, ordered
   CPU product/rescaling/log. Matrix layout [pair][rate][nCodes], JC layout
   [pair][PSame/PDiff][rate]. Outputs retain the transactional batch contract. */
typedef struct {
  FtGpuPairBatch batch;
  const double *transitions;
} FtGpuOrderedPairBatch;
FtGpuStatus ft_gpu_pair_loglk_ordered(FtGpuContext*, const FtGpuOrderedPairBatch*);


/* ---- profile distance batch (ProfileDist / SeqDist raw dist+weight) ---- */

typedef struct {
  int isSeq;                  /* 1: SeqDist semantics, 0: ProfileDist semantics */
  /* For isSeq jobs the codes can be supplied either as raw host arrays
     (codesA/codesB, copied per batch) or as code-only profiles through A/B
     (owner/serial/weights/codes with nVectors == 0), which stay resident in the
     device cache.  A/B are always used for ProfileDist jobs. */
  const unsigned char *codesA;/* [nPos] or NULL */
  const unsigned char *codesB;/* [nPos] or NULL */
  FtGpuProfileView A;
  FtGpuProfileView B;
} FtGpuDistPair;

typedef struct {
  int nPos;
  int nCodes;
  int useMatrix;                 /* 0: no distance matrix (identity/JC-ish) */
  const double *dmat_distances;  /* [FT_GPU_MAXCODES*FT_GPU_MAXCODES] (dmat->distances) */
  const double *dmat_eigenval;   /* [nCodes] */
  const double *dmat_codeFreq;   /* [FT_GPU_MAXCODES*FT_GPU_MAXCODES] (dmat->codeFreq) */
  int nPairs;
  const FtGpuDistPair *pairs;
  /* outputs: raw values, before diameter/constraint/criterion handling */
  double *out_dist;              /* [nPairs] */
  double *out_weight;            /* [nPairs] */
} FtGpuDistBatch;
/* ProfileDist only (isSeq==0): parallel site terms, ordered device reduction.
   Same transactional contract as the regular distance API. */
FtGpuStatus ft_gpu_dist_ordered(FtGpuContext*, const FtGpuDistBatch*);


typedef struct {
  uint64_t nBatches;             /* batch calls that executed on the device */
  uint64_t nPairsGpu;            /* pair jobs executed on the device */
  uint64_t nSitesGpu;            /* site evaluations (pairs * nPos) */
  uint64_t nPairsFallback;       /* pair jobs left to the CPU by the caller */
  uint64_t nKernelLaunches;
  uint64_t nProfileUploads;      /* device copies created */
  uint64_t nProfileEvictions;    /* device copies dropped by the budget */
  uint64_t nProfileHits;         /* cache hits */
  uint64_t nProfileInvalidations;/* explicit invalidations */
  uint64_t bytesH2D;
  uint64_t bytesD2H;
  uint64_t peakDeviceBytes;
  uint64_t deviceBytes;
  double tPack;                  /* index build + staging */
  double tH2D;
  double tKernel;                /* batch event time; resident uses tResident */
  double tD2H;
  double tWait;                  /* synchronization time */
  double tTotal;                 /* time inside backend calls */
  /* per-operation breakdown */
  uint64_t nPairsGpuPair;        /* PairLogLk pair jobs */
  uint64_t nPairsGpuDist;        /* distance pair jobs */
  uint64_t nSitesPair;           /* site evaluations in PairLogLk */
  uint64_t nSitesDist;           /* site evaluations in distance kernels */
  uint64_t nDistSeqJobs;         /* distance jobs that used the code/code path */
  double tKernelPair;
  double tKernelDist;
  uint64_t nResidentEvaluations;
  uint64_t nResidentNodes;
  double tResident;
} FtGpuStats;

/* ---- lifecycle ---- */

/* Returns 1 if this build has a CUDA backend and at least one usable device.
   Never touches the device beyond a cheap enumeration. */
int ft_gpu_is_available(void);

FtGpuStatus ft_gpu_create(const FtGpuConfig *config, FtGpuContext **out_ctx);
void ft_gpu_destroy(FtGpuContext *ctx);

/* Human readable device description, e.g. "NVIDIA RTX A5000 (sm_86, 24564 MiB,
   CUDA 12.6)". Returns FT_GPU_ERR_NO_DEVICE when unavailable. */
FtGpuStatus ft_gpu_device_info(FtGpuContext *ctx, char *buf, size_t buflen);

/* ---- operations ---- */

FtGpuStatus ft_gpu_pair_loglk_batch(FtGpuContext *ctx, const FtGpuPairBatch *batch);
FtGpuStatus ft_gpu_dist_batch(FtGpuContext *ctx, const FtGpuDistBatch *batch);

/* Drop any device copy associated with (owner, serial). */
FtGpuStatus ft_gpu_profile_invalidate(FtGpuContext *ctx, const void *owner, uint64_t serial);

/* Fixed-topology matrix-model phase (4 or 20 states). Children refer to
   leaves [0,nLeaves) or earlier jobs [nLeaves,nLeaves+job). The final job is
   the last root pair. Leaves are pure characters/gaps; nJobs=nLeaves-1.
   Destroy the tree before its owning context. Storage counts toward its budget.
   Calls are synchronous; failed evaluation leaves both outputs unchanged.
   Exponentials: [job][3: left posterior, right posterior, pair][rate][nCodes].
   4-state model (40 doubles): codeFreq[5][4] (gap row 4), statinv[4],
   eigeninvT[4][4]. 20-state model (1643 doubles): codeFreq[21][20] (gap row
   20), statinv[20], eigeninv[20][20], nearP[20][20], nearFreq[20][20], then
   exactML, approxMLminf, approxMLminratio. Site output is optional. */
typedef struct FtGpuResidentTree FtGpuResidentTree;
FtGpuStatus ft_gpu_ml_tree_create(FtGpuContext *ctx, int nLeaves, int nPos, int nJobs, int nCodes,
  int nRates, const unsigned char *leafCodes, const unsigned int *ratecat,
  const int *children, FtGpuResidentTree **out);
FtGpuStatus ft_gpu_ml_tree_eval(FtGpuResidentTree *tree, const double *expTables,
  const double *model, double *out_loglk, double *out_site_loglk);
void ft_gpu_ml_tree_destroy(FtGpuResidentTree *tree);

/* ---- diagnostics ---- */

void ft_gpu_stats_get(FtGpuContext *ctx, FtGpuStats *out);
void ft_gpu_stats_reset(FtGpuContext *ctx);
const char *ft_gpu_last_error(FtGpuContext *ctx);
const char *ft_gpu_status_string(FtGpuStatus status);
void ft_gpu_set_stats_level(FtGpuContext *ctx, int level);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* FT_GPU_H */
