/* SPDX-License-Identifier: GPL-3.0-or-later */
/* ft_gpu_backend.c -- backend policy, fallback routing and statistics.
 *
 * See ft_gpu_backend.h.  Compiled into both the CPU-only and the CUDA builds;
 * the CUDA-specific entry points live in ft_gpu_cuda.cu (or ft_gpu_stub.c).
 */
#include "ft_gpu_backend.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

enum { FTB_BACKEND_CPU = 0, FTB_BACKEND_CUDA = 1, FTB_BACKEND_AUTO = 2 };

static int g_backend = FTB_BACKEND_AUTO;
static int g_device = -1;
static long g_memoryMB = 0;
static int g_verify = 0;
static int g_stats = 0;
static int g_quiet = 0;
static long g_minWork = 0;   /* minimum nPairs*nPos to consider the GPU */
static long g_minPairs = 1;  /* minimum number of pairs */
static FtGpuContext *g_ctx = NULL;
static int g_initDone = 0;
static int g_initFailed = 0;
static char g_deviceInfo[256] = "";

static uint64_t g_fallbackPairs = 0;
static uint64_t g_fallbackWork = 0;
static uint64_t g_fallbackSmall = 0;
static uint64_t g_fallbackUnsupported = 0;
static uint64_t g_fallbackError = 0;
static uint64_t g_batchesOffered = 0;
static uint64_t g_pairsOffered = 0;

static void ftb_init(void);

void ftb_configure(const char *backend, int device, long memory_mb, int verify,
                   int stats, long min_work, long min_pairs) {
  /* A reconfiguration must not reuse a context for another device/budget. */
  ftb_shutdown();
  g_initDone = g_initFailed = 0;
  g_deviceInfo[0] = '\0';
  g_fallbackPairs = g_fallbackWork = g_fallbackSmall = 0;
  g_fallbackUnsupported = g_fallbackError = 0;
  g_batchesOffered = g_pairsOffered = 0;
  if (backend != NULL) {
    if (strcmp(backend, "cpu") == 0)
      g_backend = FTB_BACKEND_CPU;
    else if (strcmp(backend, "cuda") == 0)
      g_backend = FTB_BACKEND_CUDA;
    else
      g_backend = FTB_BACKEND_AUTO;
  }
  g_device = device;
  g_memoryMB = memory_mb;
  g_verify = verify;
  g_stats = stats;
  if (min_work >= 0) g_minWork = min_work;
  if (min_pairs > 0) g_minPairs = min_pairs;
  if (memory_mb < 0 || (uint64_t)memory_mb > SIZE_MAX / (1024u * 1024u)) {
    fprintf(stderr, "FastTree: invalid GPU memory budget\n");
    exit(1);
  }
  if (g_backend == FTB_BACKEND_CUDA) ftb_init(); /* after all flags are set */
}

void ftb_set_quiet(int quiet) { g_quiet = quiet; }

int ftb_cuda_requested(void) { return g_backend == FTB_BACKEND_CUDA; }
int ftb_verify(void) { return g_verify; }
int ftb_stats_enabled(void) { return g_stats; }

static void ftb_init(void) {
  g_initDone = 1;
  if (g_backend == FTB_BACKEND_CPU) {
    g_initFailed = 1;
    return;
  }
  if (!ft_gpu_is_available()) {
    if (g_backend == FTB_BACKEND_CUDA) {
      fprintf(stderr,
              "FastTree: --backend cuda was requested but this program has no usable "
              "CUDA device or was built without CUDA support\n");
      exit(1);
    }
    g_initFailed = 1;
    return;
  }
  FtGpuConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.device_id = g_device;
  cfg.memory_limit_bytes = (g_memoryMB > 0) ? (size_t)g_memoryMB * 1024u * 1024u : 0;
  cfg.verify = g_verify;
  cfg.stats_level = g_stats;
  FtGpuStatus st = ft_gpu_create(&cfg, &g_ctx);
  if (st != FT_GPU_OK) {
    if (g_backend == FTB_BACKEND_CUDA) {
      fprintf(stderr, "FastTree: --backend cuda failed: %s\n", ft_gpu_status_string(st));
      exit(1);
    }
    g_initFailed = 1;
    if (!g_quiet)
      fprintf(stderr, "# FastTree: CUDA backend unavailable (%s); running on the CPU\n",
              ft_gpu_status_string(st));
    return;
  }
  if (ft_gpu_device_info(g_ctx, g_deviceInfo, sizeof(g_deviceInfo)) != FT_GPU_OK)
    g_deviceInfo[0] = '\0';
  if (!g_quiet)
    fprintf(stderr, "# FastTree: CUDA backend enabled on %s\n",
            g_deviceInfo[0] ? g_deviceInfo : "(unknown device)");
}

int ftb_active(void) {
  if (g_backend == FTB_BACKEND_CPU) return 0;
  if (!g_initDone) ftb_init();
  return g_ctx != NULL;
}

static int ftb_worth_it(int nPairs, int nPos);

int ftb_would_use(int nJobs, int nPos) {
  if (g_backend == FTB_BACKEND_CPU) return 0;
  if (g_initFailed) return 0;
  if (!ftb_worth_it(nJobs, nPos)) return 0;
  return 1;
}

static int ftb_worth_it(int nPairs, int nPos) {
  if (nPairs <= 0 || nPos <= 0 || nPairs < g_minPairs) return 0;
  if (g_minWork > 0) {
    double work = (double)nPairs * (double)nPos;
    if (work < (double)g_minWork) return 0;
  }
  return 1;
}

void ftb_note_fallback(uint64_t nPairs, const char *reason) {
  g_fallbackPairs += nPairs;
  if (reason != NULL && strcmp(reason, "small") == 0) g_fallbackSmall += nPairs;
  else if (reason != NULL && strcmp(reason, "unsupported") == 0) g_fallbackUnsupported += nPairs;
  else if (reason != NULL && strcmp(reason, "error") == 0) g_fallbackError += nPairs;
}

static int handle_error(const char *op, FtGpuStatus st, uint64_t nPairs) {
  const char *msg = g_ctx ? ft_gpu_last_error(g_ctx) : "(no context)";
  if (st == FT_GPU_ERR_MEMORY || st == FT_GPU_ERR_UNSUPPORTED ||
      st == FT_GPU_ERR_OVERFLOW || st == FT_GPU_ERR_NO_DEVICE) {
    ftb_note_fallback(nPairs, st == FT_GPU_ERR_UNSUPPORTED ? "unsupported" : "error");
    if (!g_quiet)
      fprintf(stderr, "# FastTree: GPU %s fell back to CPU: %s (%s)\n",
              op, ft_gpu_status_string(st), msg);
    return FTB_FALLBACK;
  }
  fprintf(stderr, "FastTree: fatal GPU error in %s: %s (%s)\n",
          op, ft_gpu_status_string(st), msg);
  exit(1);
  return FTB_FATAL;
}

int ftb_pair_loglk_batch(int nPos, int nCodes, int useMatrix, const double *eigenval,
                         const double *codeFreq, int nRateCategories, const double *rates,
                         const unsigned int *ratecat, int nPairs, const FtGpuPair *pairs,
                         double *out_loglk, double *out_site) {
  if (nPairs <= 0) return FTB_FALLBACK;
  g_batchesOffered++;
  g_pairsOffered += (uint64_t)nPairs;
  /* Threshold first: a batch that would be refused must not pay for the
     one-time device initialisation either. */
  if (!ftb_worth_it(nPairs, nPos)) {
    ftb_note_fallback((uint64_t)nPairs, "small");
    g_fallbackWork += (uint64_t)nPairs * (uint64_t)nPos;
    return FTB_FALLBACK;
  }
  if (!ftb_active()) {
    ftb_note_fallback((uint64_t)nPairs, "error");
    return FTB_FALLBACK;
  }
  if ((nCodes != 4 && nCodes != 20) || (!useMatrix && nCodes != 4) ||
      nRateCategories > FT_GPU_MAX_RATE_CATEGORIES) {
    ftb_note_fallback((uint64_t)nPairs, "unsupported");
    return FTB_FALLBACK;
  }
  FtGpuPairBatch batch;
  memset(&batch, 0, sizeof(batch));
  batch.nPos = nPos;
  batch.model.nCodes = nCodes;
  batch.model.useMatrix = useMatrix;
  batch.model.eigenval = eigenval;
  batch.model.codeFreq = codeFreq;
  batch.rates.nRateCategories = nRateCategories;
  batch.rates.rates = rates;
  batch.rates.ratecat = ratecat;
  batch.nPairs = nPairs;
  batch.pairs = pairs;
  batch.out_loglk = out_loglk;
  batch.out_site = out_site;
  FtGpuStatus st = ft_gpu_pair_loglk_batch(g_ctx, &batch);
  if (st == FT_GPU_OK) return FTB_DONE;
  return handle_error("PairLogLk batch", st, (uint64_t)nPairs);
}

int ftb_dist_batch(int nPos, int nCodes, int useMatrix, const double *dmatDistances,
                   const double *dmatEigenval, const double *dmatCodeFreq, int nPairs,
                   const FtGpuDistPair *pairs, double *outDist, double *outWeight) {
  if (nPairs <= 0) return FTB_FALLBACK;
  g_batchesOffered++;
  g_pairsOffered += (uint64_t)nPairs;
  if (!ftb_worth_it(nPairs, nPos)) {
    ftb_note_fallback((uint64_t)nPairs, "small");
    g_fallbackWork += (uint64_t)nPairs * (uint64_t)nPos;
    return FTB_FALLBACK;
  }
  if (!ftb_active()) {
    ftb_note_fallback((uint64_t)nPairs, "error");
    return FTB_FALLBACK;
  }
  FtGpuDistBatch batch;
  memset(&batch, 0, sizeof(batch));
  batch.nPos = nPos;
  batch.nCodes = nCodes;
  batch.useMatrix = useMatrix;
  batch.dmat_distances = dmatDistances;
  batch.dmat_eigenval = dmatEigenval;
  batch.dmat_codeFreq = dmatCodeFreq;
  batch.nPairs = nPairs;
  batch.pairs = pairs;
  batch.out_dist = outDist;
  batch.out_weight = outWeight;
  FtGpuStatus st = ft_gpu_dist_batch(g_ctx, &batch);
  if (st == FT_GPU_OK) return FTB_DONE;
  return handle_error("distance batch", st, (uint64_t)nPairs);
}

void ftb_profile_invalidate(const void *owner, uint64_t serial) {
  if (g_ctx == NULL) return;
  (void)ft_gpu_profile_invalidate(g_ctx, owner, serial);
}

int ftb_verify_doubles(const char *what, const double *gpu, const double *cpu, int n) {
  int i;
  int nBad = 0;
  double worst = 0.0;
  int worstIdx = -1;
  for (i = 0; i < n; i++) {
    double diff = fabs(gpu[i] - cpu[i]);
    double tol = 1e-12 + 1e-10 * fabs(cpu[i]);
    if (!isfinite(gpu[i]) || !isfinite(cpu[i]) || !(diff <= tol)) { /* also catches NaN */
      nBad++;
      if (worstIdx < 0 || diff - tol > worst) {
        worst = diff - tol;
        worstIdx = i;
      }
    }
  }
  if (nBad > 0) {
    fprintf(stderr,
            "FastTree: --gpu-verify FAILED for %s: %d of %d values outside "
            "1e-12 + 1e-10*|cpu| (worst at index %d: gpu=%.17g cpu=%.17g)\n",
            what, nBad, n, worstIdx, gpu[worstIdx], cpu[worstIdx]);
    return 0;
  }
  return 1;
}

void ftb_report_device(FILE *fp) {
  if (!ftb_active()) {
    fprintf(fp, "# GPU backend: not active (%s)\n",
            g_backend == FTB_BACKEND_CPU ? "disabled by --backend cpu" : "no usable device");
    return;
  }
  fprintf(fp, "# GPU backend: %s\n", g_deviceInfo[0] ? g_deviceInfo : "(unknown device)");
}

void ftb_report_stats(FILE *fp) {
  FtGpuStats s;
  memset(&s, 0, sizeof(s));
  if (g_ctx != NULL) ft_gpu_stats_get(g_ctx, &s);
  fprintf(fp, "# GPU stats: backend=%s verify=%d\n",
          g_backend == FTB_BACKEND_CPU ? "cpu" : (g_backend == FTB_BACKEND_CUDA ? "cuda" : "auto"),
          g_verify);
  fprintf(fp, "# GPU stats: device=%s\n", g_deviceInfo[0] ? g_deviceInfo : "none");
  fprintf(fp, "# GPU stats: kernel_launches=%llu batches=%llu pair_jobs=%llu site_evals=%llu\n",
          (unsigned long long)s.nKernelLaunches, (unsigned long long)s.nBatches,
          (unsigned long long)s.nPairsGpu, (unsigned long long)s.nSitesGpu);
  fprintf(fp,
          "# GPU stats: profile_uploads=%llu cache_hits=%llu evictions=%llu invalidations=%llu\n",
          (unsigned long long)s.nProfileUploads, (unsigned long long)s.nProfileHits,
          (unsigned long long)s.nProfileEvictions, (unsigned long long)s.nProfileInvalidations);
  fprintf(fp, "# GPU stats: bytes_h2d=%llu bytes_d2h=%llu device_bytes=%llu peak_device_bytes=%llu\n",
          (unsigned long long)s.bytesH2D, (unsigned long long)s.bytesD2H,
          (unsigned long long)s.deviceBytes, (unsigned long long)s.peakDeviceBytes);
  fprintf(fp,
          "# GPU stats: t_pack=%.6f t_h2d=%.6f t_kernel=%.6f t_wait=%.6f t_d2h=%.6f "
          "t_backend_total=%.6f (s)\n",
          s.tPack, s.tH2D, s.tKernel, s.tWait, s.tD2H, s.tTotal);
  fprintf(fp,
          "# GPU stats: pairloglk_jobs=%llu pairloglk_sites=%llu pairloglk_kernel=%.6f s | "
          "dist_jobs=%llu (of which code/code=%llu) dist_sites=%llu dist_kernel=%.6f s\n",
          (unsigned long long)s.nPairsGpuPair, (unsigned long long)s.nSitesPair,
          s.tKernelPair, (unsigned long long)s.nPairsGpuDist,
          (unsigned long long)s.nDistSeqJobs, (unsigned long long)s.nSitesDist, s.tKernelDist);
  fprintf(fp,
          "# GPU stats: fallback_pairs=%llu (small_or_threshold=%llu unsupported=%llu "
          "error=%llu) batches_offered=%llu pairs_offered=%llu\n",
          (unsigned long long)(g_fallbackSmall + g_fallbackUnsupported + g_fallbackError),
          (unsigned long long)g_fallbackSmall, (unsigned long long)g_fallbackUnsupported,
          (unsigned long long)g_fallbackError, (unsigned long long)g_batchesOffered,
          (unsigned long long)g_pairsOffered);
  fprintf(fp, "# GPU resident: evaluations=%llu posterior_nodes=%llu wall_seconds=%.6f\n",
          (unsigned long long)s.nResidentEvaluations, (unsigned long long)s.nResidentNodes, s.tResident);
  fprintf(fp, "# GPU stats: thresholds min_work=%ld(site-pairs) min_pairs=%ld memory_mb=%ld\n",
          g_minWork, g_minPairs, g_memoryMB);
}

void ftb_shutdown(void) {
  if (g_ctx != NULL) {
    ft_gpu_destroy(g_ctx);
    g_ctx = NULL;
  }
  g_initDone = 1;
  g_initFailed = 1;
}

FtGpuContext *ftb_context(void) { return ftb_active() ? g_ctx : NULL; }
