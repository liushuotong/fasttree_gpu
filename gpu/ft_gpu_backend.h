/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ft_gpu_backend.h -- CPU-side dispatch layer between FastTree.c and the
 * optional CUDA backend.  This header is included by FastTree.c; it must stay
 * plain C and must not expose CUDA or C++ types.
 *
 * Responsibilities:
 *  - lazy context creation and backend policy (cpu / cuda / auto)
 *  - profitability thresholds and CPU fallback bookkeeping
 *  - verification of GPU results against the CPU implementation
 *  - statistics reporting
 */
#ifndef FT_GPU_BACKEND_H
#define FT_GPU_BACKEND_H

#include <stdio.h>
#include <stdint.h>

#include "ft_gpu.h"

/* Default minimum batch size (pairs * sites) before a batch is sent to the
   device.  Calibrated on an RTX A5000 (see docs/IMPLEMENTATION.md); override with
   -gpu-min-work.  Below this the launch/transfer latency is not recovered. */
#define FTB_DEFAULT_MIN_WORK 200000L

/* return codes of the batch helpers */
#define FTB_FALLBACK 0 /* caller must use its CPU path */
#define FTB_DONE 1     /* results are in the output arrays */
#define FTB_FATAL (-1) /* unrecoverable; the caller must stop */

void ftb_configure(const char *backend, int device, long memory_mb, int verify,
                   int stats, long min_work, long min_pairs);
void ftb_set_quiet(int quiet);

/* 1 if the CUDA backend is usable (performs lazy context creation once).
   Fast enough to call in hot loops: it is a cached flag test afterwards. */
int ftb_active(void);
FtGpuContext *ftb_context(void);

/* Cheap, side-effect-free test: would a batch of nJobs x nPos be routed to the
   device?  It never creates a context, so callers can skip building a batch
   (and the one-time ~0.15 s device initialisation) when the answer is no. */
int ftb_would_use(int nJobs, int nPos);

/* 1 if the user asked for --backend cuda (so fallbacks must be reported) */
int ftb_cuda_requested(void);
int ftb_verify(void);
int ftb_stats_enabled(void);

/* Batch PairLogLk.  On FTB_DONE, out_loglk[0..nPairs-1] are filled and
   out_site (if not NULL) holds the per-site lkAB values; the caller applies
   site_likelihoods[i] *= out_site[i] itself so that the CPU semantics and the
   commit point stay under the caller's control. */
int ftb_pair_loglk_batch(int nPos, int nCodes, int useMatrix,
                         const double *eigenval, const double *codeFreq,
                         int nRateCategories, const double *rates,
                         const unsigned int *ratecat, int nPairs,
                         const FtGpuPair *pairs, double *out_loglk, double *out_site);

/* Batch raw profile/sequence distances. */
int ftb_dist_batch(int nPos, int nCodes, int useMatrix, const double *dmatDistances,
                   const double *dmatEigenval, const double *dmatCodeFreq, int nPairs,
                   const FtGpuDistPair *pairs, double *outDist, double *outWeight);

/* Drop the device copy of a profile that is about to be freed or rewritten. */
void ftb_profile_invalidate(const void *owner, uint64_t serial);

/* Bookkeeping for pairs the caller decided to keep on the CPU. */
void ftb_note_fallback(uint64_t nPairs, const char *reason);

/* Compare a GPU result array against the CPU reference; returns 1 if within
   tolerance, 0 otherwise (details are printed once per site). */
int ftb_verify_doubles(const char *what, const double *gpu, const double *cpu, int n);

void ftb_report_stats(FILE *fp);
void ftb_report_device(FILE *fp);
void ftb_shutdown(void);

#endif /* FT_GPU_BACKEND_H */
