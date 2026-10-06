/* bench_kernels.c -- throughput and crossover measurements for the CUDA
 * backend kernels, plus the CPU reference on the same work.
 *
 * Built the same way as tests/test_kernels.c: it includes FastTree.c so that
 * the CPU reference kernel and the real model code are available.
 *
 *   bench_kernels --mode pair|dist --pairs N --sites L --codes 4|20 --repeats R
 *                 [--model jc|gtr|jtt] [--csv]
 *
 * The report separates kernel time (CUDA events), host-to-device and
 * device-to-host time (events in stream order), synchronization overhead and
 * the wall time of the whole batch call, so the fixed per-batch cost can be
 * read off directly from the nPairs = 1 rows.
 */
#define main fasttree_main
#include "../src/FastTree.c"
#undef main

#include "profile_pack.h"

#include <stdint.h>

typedef struct {
  double *weights;
  unsigned char *codes;
  double *vectors;
  int nVectors;
  profile_t profile;
} BP;

static uint64_t rng = 12345678901234567ULL;
static double rnd01(void) {
  rng ^= rng << 13;
  rng ^= rng >> 7;
  rng ^= rng << 17;
  return (double)(rng >> 11) / 9007199254740992.0;
}

static void bp_build(BP *p, int nPos, int nc, const numeric_t (*codeFreq)[MAXCODES]) {
  int i, k;
  memset(p, 0, sizeof(*p));
  p->weights = (double *)malloc(sizeof(double) * (size_t)nPos);
  p->codes = (unsigned char *)malloc((size_t)nPos);
  for (i = 0; i < nPos; i++) {
    if (rnd01() < 0.05) { /* all-gap site */
      p->weights[i] = 0.0;
      p->codes[i] = NOCODE;
    } else {
      p->weights[i] = 1.0;
      p->codes[i] = NOCODE;
    }
  }
  p->nVectors = ft_count_vectors(p->weights, p->codes, nPos);
  if (p->nVectors < 0) p->nVectors = 0;
  if (p->nVectors > 0)
    p->vectors = (double *)malloc(sizeof(double) * (size_t)p->nVectors * nc);
  {
    int v = 0;
    for (i = 0; i < nPos; i++) {
      double *f;
      if (!FT_HAS_VECTOR(p->weights, p->codes, i)) continue;
      f = &p->vectors[(size_t)v * nc];
      if (codeFreq != NULL) {
        double alpha[MAXCODES];
        double sum = 0.0;
        int r;
        for (r = 0; r < nc; r++) {
          alpha[r] = 0.05 + rnd01();
          sum += alpha[r];
        }
        for (k = 0; k < nc; k++) f[k] = 0.0;
        for (r = 0; r < nc; r++)
          for (k = 0; k < nc; k++) f[k] += (alpha[r] / sum) * codeFreq[r][k];
      } else {
        double sum = 0.0;
        for (k = 0; k < nc; k++) {
          f[k] = 0.02 + rnd01();
          sum += f[k];
        }
        for (k = 0; k < nc; k++) f[k] /= sum;
      }
      v++;
    }
  }
  p->profile.weights = p->weights;
  p->profile.codes = p->codes;
  p->profile.vectors = p->vectors;
  p->profile.nVectors = p->nVectors;
  p->profile.codeDist = NULL;
  p->profile.nOn = NULL;
  p->profile.nOff = NULL;
  p->profile.ftSerial = ++rng;
}

static FtGpuProfileView bp_view(BP *p) {
  FtGpuProfileView v;
  memset(&v, 0, sizeof(v));
  v.owner = &p->profile;
  v.serial = p->profile.ftSerial;
  v.weights = p->weights;
  v.codes = p->codes;
  v.vectors = p->vectors;
  v.nVectors = p->nVectors;
  return v;
}

static double now_s(void) {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char **argv) {
  const char *mode = "pair";
  const char *modelName = "jtt";
  int nPairs = 100;
  int nPos = 1000;
  int nc = 20;
  int nRepeats = 3;
  int csv = 0;
  int seqMode = 0;
  int i, r;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
    else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) modelName = argv[++i];
    else if (strcmp(argv[i], "--pairs") == 0 && i + 1 < argc) nPairs = atoi(argv[++i]);
    else if (strcmp(argv[i], "--sites") == 0 && i + 1 < argc) nPos = atoi(argv[++i]);
    else if (strcmp(argv[i], "--codes") == 0 && i + 1 < argc) nc = atoi(argv[++i]);
    else if (strcmp(argv[i], "--repeats") == 0 && i + 1 < argc) nRepeats = atoi(argv[++i]);
    else if (strcmp(argv[i], "--csv") == 0) csv = 1;
    else if (strcmp(argv[i], "--seq") == 0) seqMode = 1;
    else {
      fprintf(stderr, "bench_kernels: unknown option %s\n", argv[i]);
      return 2;
    }
  }
  showProgress = 0;
  nCodes = nc;
  codesString = (nc == 20) ? codesStringAA : codesStringNT;

  transition_matrix_t *tm = NULL;
  distance_matrix_t *dmat = NULL;
  int useMatrix = 1;
  if (nc == 20) {
    tm = CreateTransitionMatrix(matrixJTT92, statJTT92);
    dmat = TransMatToDistanceMat(tm);
  } else if (strcmp(modelName, "jc") == 0) {
    useMatrix = 0;
  } else {
    double gtrrates[6] = {1.0, 2.0, 1.0, 1.0, 2.0, 1.0};
    double gtrfreq[4] = {0.25, 0.25, 0.25, 0.25};
    tm = CreateGTR(gtrrates, gtrfreq);
    dmat = TransMatToDistanceMat(tm);
  }

  rates_t rates;
  rates.nRateCategories = 20;
  rates.rates = MLSiteRates(20);
  rates.ratecat = (unsigned int *)malloc(sizeof(unsigned int) * (size_t)nPos);
  for (i = 0; i < nPos; i++) rates.ratecat[i] = (unsigned int)(i % 20);

  BP *As = (BP *)malloc(sizeof(BP) * (size_t)nPairs);
  BP *Bs = (BP *)malloc(sizeof(BP) * (size_t)nPairs);
  FtGpuPair *pairs = (FtGpuPair *)malloc(sizeof(FtGpuPair) * (size_t)nPairs);
  FtGpuDistPair *dpairs = (FtGpuDistPair *)malloc(sizeof(FtGpuDistPair) * (size_t)nPairs);
  double *outLk = (double *)malloc(sizeof(double) * (size_t)nPairs);
  double *outDist = (double *)malloc(sizeof(double) * (size_t)nPairs);
  double *outWeight = (double *)malloc(sizeof(double) * (size_t)nPairs);

  for (i = 0; i < nPairs; i++) {
    bp_build(&As[i], nPos, nc, tm ? tm->codeFreq : NULL);
    bp_build(&Bs[i], nPos, nc, tm ? tm->codeFreq : NULL);
    memset(&pairs[i], 0, sizeof(FtGpuPair));
    pairs[i].A = bp_view(&As[i]);
    pairs[i].B = bp_view(&Bs[i]);
    pairs[i].length = 0.05 + 0.001 * (i % 100);
    memset(&dpairs[i], 0, sizeof(FtGpuDistPair));
    dpairs[i].isSeq = seqMode;
    if (seqMode) {
      /* leaf-like: codes only, no vectors */
      dpairs[i].codesA = As[i].codes;
      dpairs[i].codesB = Bs[i].codes;
    } else {
      dpairs[i].A = bp_view(&As[i]);
      dpairs[i].B = bp_view(&Bs[i]);
    }
  }

  /* CPU reference on the same batch */
  double cpuWall = now_s();
  if (strcmp(mode, "pair") == 0) {
    for (i = 0; i < nPairs; i++)
      outLk[i] = PairLogLk(&As[i].profile, &Bs[i].profile, pairs[i].length, nPos,
                           useMatrix ? tm : NULL, &rates, NULL);
  } else if (seqMode) {
    besthit_t hit;
    for (i = 0; i < nPairs; i++) {
      SeqDist(As[i].codes, Bs[i].codes, nPos, dmat, &hit);
      outDist[i] = hit.dist;
      outWeight[i] = hit.weight;
    }
  } else {
    besthit_t hit;
    for (i = 0; i < nPairs; i++) {
      ProfileDist(&As[i].profile, &Bs[i].profile, nPos, dmat, &hit);
      outDist[i] = hit.dist;
      outWeight[i] = hit.weight;
    }
  }
  cpuWall = now_s() - cpuWall;

  FtGpuContext *ctx = NULL;
  FtGpuConfig cfg;
  memset(&cfg, 0, sizeof(cfg));
  cfg.device_id = -1;
  cfg.stats_level = 2;
  FtGpuStatus st = ft_gpu_create(&cfg, &ctx);
  if (st != FT_GPU_OK) {
    printf("no CUDA device: %s\n", ft_gpu_status_string(st));
    return 1;
  }

  double bestWall = 1e30, bestKernel = 0, bestH2D = 0, bestD2H = 0, bestWait = 0;
  FtGpuStats s;
  for (r = 0; r < nRepeats; r++) {
    ft_gpu_stats_reset(ctx);
    double wall = now_s();
    if (strcmp(mode, "pair") == 0) {
      FtGpuPairBatch batch;
      memset(&batch, 0, sizeof(batch));
      batch.nPos = nPos;
      batch.model.nCodes = nc;
      batch.model.useMatrix = useMatrix;
      batch.model.eigenval = tm ? tm->eigenval : NULL;
      batch.model.codeFreq = tm ? &tm->codeFreq[0][0] : NULL;
      batch.rates.nRateCategories = rates.nRateCategories;
      batch.rates.rates = rates.rates;
      batch.rates.ratecat = rates.ratecat;
      batch.nPairs = nPairs;
      batch.pairs = pairs;
      batch.out_loglk = outLk;
      batch.out_site = NULL;
      st = ft_gpu_pair_loglk_batch(ctx, &batch);
    } else {
      FtGpuDistBatch batch;
      memset(&batch, 0, sizeof(batch));
      batch.nPos = nPos;
      batch.nCodes = nc;
      batch.useMatrix = (dmat != NULL);
      batch.dmat_distances = dmat ? &dmat->distances[0][0] : NULL;
      batch.dmat_eigenval = dmat ? dmat->eigenval : NULL;
      batch.dmat_codeFreq = dmat ? &dmat->codeFreq[0][0] : NULL;
      batch.nPairs = nPairs;
      batch.pairs = dpairs;
      batch.out_dist = outDist;
      batch.out_weight = outWeight;
      st = ft_gpu_dist_batch(ctx, &batch);
    }
    wall = now_s() - wall;
    if (st != FT_GPU_OK) {
      printf("backend error: %s (%s)\n", ft_gpu_status_string(st), ft_gpu_last_error(ctx));
      return 1;
    }
    ft_gpu_stats_get(ctx, &s);
    if (wall < bestWall) {
      bestWall = wall;
      bestKernel = s.tKernel;
      bestH2D = s.tH2D;
      bestD2H = s.tD2H;
      bestWait = s.tWait;
    }
  }

  double siteEvals = (double)nPairs * (double)nPos;
  double perSiteFlops = (strcmp(mode, "pair") == 0)
                            ? (2.0 * nc + 4.0)
                            : (seqMode ? 3.0 : (2.0 * nc + 3.0));
  if (csv) {
    printf("%s,%s,%d,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.1f,%.2f\n", mode, modelName, nc, nPairs,
           nPos, cpuWall, bestWall, bestH2D, bestKernel, bestD2H, bestWait,
           siteEvals / bestKernel / 1e9, siteEvals * perSiteFlops / bestKernel / 1e9);
  } else {
    printf("mode=%s model=%s codes=%d pairs=%d sites=%d | CPU %.4f s | GPU wall %.4f s "
           "(h2d %.4f kernel %.4f d2h %.4f wait %.4f) | %.2f Gsite/s kernel, %.1f GFLOP/s "
           "kernel | speedup vs CPU %.2fx\n",
           mode, modelName, nc, nPairs, nPos, cpuWall, bestWall, bestH2D, bestKernel, bestD2H,
           bestWait, siteEvals / bestKernel / 1e9, siteEvals * perSiteFlops / bestKernel / 1e9,
           cpuWall / bestWall);
  }
  ft_gpu_destroy(ctx);
  return 0;
}
