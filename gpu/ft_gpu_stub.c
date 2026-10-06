/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ft_gpu_stub.c -- backend used when the project is built without CUDA.
 *
 * It satisfies the same C ABI as the CUDA backend so that FastTree.c can link
 * and run unchanged, and every operation reports FT_GPU_ERR_NO_DEVICE.  The
 * dispatch layer turns that into a CPU fallback.
 */
#include "ft_gpu.h"

#include <string.h>

int ft_gpu_is_available(void) { return 0; }

FtGpuStatus ft_gpu_create(const FtGpuConfig *config, FtGpuContext **out_ctx) {
  (void)config;
  if (out_ctx != NULL) *out_ctx = NULL;
  return FT_GPU_ERR_NO_DEVICE;
}

void ft_gpu_destroy(FtGpuContext *ctx) { (void)ctx; }

FtGpuStatus ft_gpu_device_info(FtGpuContext *ctx, char *buf, size_t buflen) {
  (void)ctx;
  if (buf != NULL && buflen > 0) {
    const char *msg = "no CUDA backend in this build";
    size_t n = strlen(msg);
    if (n >= buflen) n = buflen - 1;
    memcpy(buf, msg, n);
    buf[n] = '\0';
  }
  return FT_GPU_ERR_NO_DEVICE;
}

FtGpuStatus ft_gpu_pair_loglk_batch(FtGpuContext *ctx, const FtGpuPairBatch *batch) {
  (void)ctx;
  (void)batch;
  return FT_GPU_ERR_NO_DEVICE;
}

FtGpuStatus ft_gpu_dist_batch(FtGpuContext *ctx, const FtGpuDistBatch *batch) {
  (void)ctx;
  (void)batch;
  return FT_GPU_ERR_NO_DEVICE;
}

FtGpuStatus ft_gpu_profile_invalidate(FtGpuContext *ctx, const void *owner, uint64_t serial) {
  (void)ctx;
  (void)owner;
  (void)serial;
  return FT_GPU_ERR_NO_DEVICE;
}

void ft_gpu_stats_get(FtGpuContext *ctx, FtGpuStats *out) {
  (void)ctx;
  if (out != NULL) memset(out, 0, sizeof(*out));
}

void ft_gpu_stats_reset(FtGpuContext *ctx) { (void)ctx; }

const char *ft_gpu_last_error(FtGpuContext *ctx) {
  (void)ctx;
  return "no CUDA backend in this build";
}

const char *ft_gpu_status_string(FtGpuStatus status) {
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

void ft_gpu_set_stats_level(FtGpuContext *ctx, int level) {
  (void)ctx;
  (void)level;
}

FtGpuStatus ft_gpu_ml_tree_create(FtGpuContext *ctx, int nLeaves, int nPos, int nJobs, int nCodes,
 int nRates, const unsigned char *codes, const unsigned int *ratecat,
 const int *children, FtGpuResidentTree **out) {
 (void)ctx;(void)nLeaves;(void)nPos;(void)nJobs;(void)nCodes;(void)nRates;
 (void)codes;(void)ratecat;(void)children;if(out)*out=NULL;
 return FT_GPU_ERR_NO_DEVICE;
}
FtGpuStatus ft_gpu_ml_tree_eval(FtGpuResidentTree *tree, const double *expTables,
 const double *model, double *out, double *out_site) {
 (void)tree;(void)expTables;(void)model;(void)out;(void)out_site;return FT_GPU_ERR_NO_DEVICE;
}
void ft_gpu_ml_tree_destroy(FtGpuResidentTree *tree) {(void)tree;}

FtGpuStatus ft_gpu_jc_posterior(FtGpuContext *ctx, int n, int nr,
 const FtGpuProfileView *a, const FtGpuProfileView *b,
 const unsigned int *cat, const double *table,
 double *weights, unsigned char *codes, double *vectors) {
 (void)ctx;(void)n;(void)nr;(void)a;(void)b;(void)cat;(void)table;
 (void)weights;(void)codes;(void)vectors;return FT_GPU_ERR_NO_DEVICE;
}

FtGpuStatus ft_gpu_pair_loglk_ordered(FtGpuContext *ctx,const FtGpuOrderedPairBatch *b){(void)ctx;(void)b;return FT_GPU_ERR_NO_DEVICE;}

FtGpuStatus ft_gpu_dist_ordered(FtGpuContext *ctx,const FtGpuDistBatch *b){(void)ctx;(void)b;return FT_GPU_ERR_NO_DEVICE;}
