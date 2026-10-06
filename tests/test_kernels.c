/* test_kernels.c -- differential tests for the FastTree CUDA backend.
 *
 * The test is built by including FastTree.c itself (with main renamed), so it
 * can use the real profile constructors, transition matrices, rate models and
 * the reference CPU kernels.  Nothing in this file is part of the production
 * program.
 *
 * Usage:
 *   test_kernels [--alignment FILE] [--seed N] [--list] [--verbose]
 *
 * Exit status is 0 only if every executed case passed.  Cases that need a GPU
 * are skipped (and reported as skipped) when no device is available, so the
 * same binary doubles as a CPU-only regression test.
 */
#define main fasttree_main
#include "../src/FastTree.c"
#undef main

#include "profile_pack.h"

#include <stdarg.h>

/* ------------------------------------------------------------------ state */

static int g_pass = 0;
static int g_fail = 0;
static int g_skip = 0;
static int g_list = 0;
static int g_verbose = 0;
static int g_realNt = 0;
static int g_seqRef = 0;
static const char *g_case = "";

static double g_worstScalar = 0.0;
static double g_worstScalarRel = 0.0;
static double g_worstSite = 0.0;
static long g_nScalarChecks = 0, g_nSiteChecks = 0, g_nDistChecks = 0;

static FtGpuContext *g_ctx = NULL;

static void case_begin(const char *name) {
  g_case = name;
  if (g_list) printf("case: %s\n", name);
}

static void case_pass(void) { g_pass++; }

static void case_fail(const char *fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  fprintf(stderr, "FAIL [%s]: ", g_case);
  vfprintf(stderr, fmt, ap);
  fprintf(stderr, "\n");
  va_end(ap);
  g_fail++;
}

static void case_skip(const char *why) {
  g_skip++;
  if (g_list) printf("  skipped: %s\n", why);
}

/* Tolerance recommended by FASTTREE_CUDA_HANDOFF.md section 11.2. */
static double tol_of(double cpu) { return 1e-12 + 1e-10 * fabs(cpu); }

static void check_scalar(const char *what, double gpu, double cpu) {
  double tol = tol_of(cpu);
  double d = fabs(gpu - cpu);
  double rel = d / (fabs(cpu) > 0 ? fabs(cpu) : 1.0);
  g_nScalarChecks++;
  if (d / tol > g_worstScalar) g_worstScalar = d / tol;
  if (rel > g_worstScalarRel) g_worstScalarRel = rel;
  if (!(d <= tol))
    case_fail("%s: gpu=%.17g cpu=%.17g diff=%.3g rel=%.3g tol=%.3g (%.1fx tol)", what, gpu,
              cpu, d, rel, tol, d / tol);
}

static void check_site(const char *what, const double *gpu, const double *cpu, int n) {
  int i;
  int nBad = 0;
  int first = -1;
  double worst = 0.0;
  for (i = 0; i < n; i++) {
    double tol = tol_of(cpu[i]);
    double d = fabs(gpu[i] - cpu[i]);
    g_nSiteChecks++;
    if (d / tol > g_worstSite) g_worstSite = d / tol;
    if (!(d <= tol)) {
      if (first < 0) first = i;
      nBad++;
      if (d / tol > worst) worst = d / tol;
    }
  }
  if (nBad > 0)
    case_fail("%s: %d/%d sites outside tolerance (first index %d: gpu=%.17g cpu=%.17g, %.1fx tol)",
              what, nBad, n, first, gpu[first], cpu[first], worst);
}

/* -------------------------------------------------------------------- rng */

static uint64_t g_rng = 88172645463325252ULL;

static double rnd01(void) {
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 7;
  g_rng ^= g_rng << 17;
  return (double)(g_rng >> 11) / 9007199254740992.0;
}

static double rnd_signed(void) { return 2.0 * rnd01() - 1.0; }

/* --------------------------------------------------------------- profiles */

typedef struct {
  int nPos;
  int nCodes;
  double *weights;
  unsigned char *codes;
  double *vectors;
  int nVectors;
  int ownsArrays;
  profile_t profile;
} P;

enum { SITE_CODE = 0, SITE_PARTIAL = 1, SITE_VECTOR = 2, SITE_GAP = 3 };
enum { STYLE_CODES = 0, STYLE_VECTORS = 1, STYLE_MIXED = 2, STYLE_GAPPY = 3 };

static void P_init(P *p, int nPos, int nCodesLocal) {
  memset(p, 0, sizeof(*p));
  p->nPos = nPos;
  p->nCodes = nCodesLocal;
  p->weights = (double *)malloc(sizeof(double) * (size_t)nPos);
  p->codes = (unsigned char *)malloc((size_t)nPos);
  p->ownsArrays = 1;
}

/* Builds a profile whose site kinds and state vectors follow the requested
   style.

   vectorSource selects how the stored state vectors are produced:
     NULL          -- positive, normalized "probability-like" vectors
     codeFreqRows  -- a random convex combination of the codeFreq rows of a
                      real model/distance matrix.  This is what FastTree's
                      AddToFreq()/posterior code produces, so the vectors are
                      legitimate eigenspace vectors and generally contain
                      negative components.  gapRow != 0 also mixes in the
                      NOCODE row (valid for transition matrices).            */
static void P_build2(P *p, int nPos, int nCodesLocal, int style,
                     const numeric_t (*codeFreqRows)[MAXCODES], int gapRow) {
  int i, k;
  int *kind = (int *)malloc(sizeof(int) * (size_t)nPos);
  P_init(p, nPos, nCodesLocal);
  for (i = 0; i < nPos; i++) {
    switch (style) {
      case STYLE_CODES:
        kind[i] = (rnd01() < 0.10) ? SITE_GAP : SITE_CODE;
        break;
      case STYLE_VECTORS:
        kind[i] = SITE_VECTOR;
        break;
      case STYLE_GAPPY:
        kind[i] = (rnd01() < 0.35) ? SITE_GAP : SITE_CODE;
        break;
      default: {
        double r = rnd01();
        kind[i] = (r < 0.35) ? SITE_CODE
                             : (r < 0.55) ? SITE_PARTIAL : (r < 0.9) ? SITE_VECTOR : SITE_GAP;
        break;
      }
    }
    if (kind[i] == SITE_CODE) {
      int c = (int)(rnd01() * nCodesLocal);
      if (c >= nCodesLocal) c = nCodesLocal - 1;
      p->codes[i] = (unsigned char)c;
      p->weights[i] = 1.0;
    } else if (kind[i] == SITE_PARTIAL) {
      p->codes[i] = NOCODE;
      p->weights[i] = 0.05 + 0.9 * rnd01();
      if (p->weights[i] >= 1.0) p->weights[i] = 0.999;
    } else if (kind[i] == SITE_VECTOR) {
      p->codes[i] = NOCODE;
      p->weights[i] = 1.0;
    } else {
      p->codes[i] = NOCODE;
      p->weights[i] = 0.0;
    }
  }
  p->nVectors = ft_count_vectors(p->weights, p->codes, nPos);
  if (p->nVectors < 0) p->nVectors = 0;
  if (p->nVectors > 0)
    p->vectors = (double *)malloc(sizeof(double) * (size_t)p->nVectors * nCodesLocal);
  {
    int v = 0;
    for (i = 0; i < nPos; i++) {
      double *f;
      if (!FT_HAS_VECTOR(p->weights, p->codes, i)) continue;
      f = &p->vectors[(size_t)v * nCodesLocal];
      if (codeFreqRows == NULL) {
        double sum = 0.0;
        for (k = 0; k < nCodesLocal; k++) {
          f[k] = 0.02 + rnd01();
          sum += f[k];
        }
        for (k = 0; k < nCodesLocal; k++) f[k] /= sum;
      } else {
        double alpha[MAXCODES + 1];
        double sum = 0.0;
        int nRows = nCodesLocal + (gapRow ? 1 : 0);
        int r;
        for (r = 0; r < nCodesLocal; r++) {
          alpha[r] = 0.05 + rnd01();
          sum += alpha[r];
        }
        if (gapRow) {
          alpha[nCodesLocal] = 0.1 * rnd01();
          sum += alpha[nCodesLocal];
        }
        for (k = 0; k < nCodesLocal; k++) f[k] = 0.0;
        for (r = 0; r < nRows; r++) {
          int row = (r == nCodesLocal) ? NOCODE : r;
          double w = alpha[r] / sum;
          for (k = 0; k < nCodesLocal; k++) f[k] += w * codeFreqRows[row][k];
        }
      }
      v++;
    }
  }
  if (p->nVectors == 0) {
    free(p->vectors);
    p->vectors = NULL;
  }
  p->profile.weights = p->weights;
  p->profile.codes = p->codes;
  p->profile.vectors = p->vectors;
  p->profile.nVectors = p->nVectors;
  p->profile.codeDist = NULL;
  p->profile.nOn = NULL;
  p->profile.nOff = NULL;
  p->profile.ftSerial = ++g_rng; /* any unique value; kept non-zero */
  free(kind);
}

/* All-gap profile: no vectors at all. */
static void P_build_empty(P *p, int nPos, int nCodesLocal) {
  int i;
  P_init(p, nPos, nCodesLocal);
  for (i = 0; i < nPos; i++) {
    p->weights[i] = 0.0;
    p->codes[i] = NOCODE;
  }
  p->nVectors = 0;
  p->vectors = NULL;
  p->profile.weights = p->weights;
  p->profile.codes = p->codes;
  p->profile.vectors = NULL;
  p->profile.nVectors = 0;
  p->profile.codeDist = NULL;
  p->profile.nOn = NULL;
  p->profile.nOff = NULL;
  p->profile.ftSerial = ++g_rng;
}

/* Read-only view of an existing profile_t (used for profiles that FastTree
   allocated itself). */
static void P_wrap(P *p, profile_t *src) {
  memset(p, 0, sizeof(*p));
  p->nPos = 0;
  p->nCodes = nCodes;
  p->weights = src->weights;
  p->codes = src->codes;
  p->vectors = src->vectors;
  p->nVectors = src->nVectors;
  p->ownsArrays = 0;
  p->profile = *src;
}

static void P_free(P *p) {
  if (p->ownsArrays) {
    free(p->weights);
    free(p->codes);
    free(p->vectors);
  }
  p->weights = NULL;
  p->codes = NULL;
  p->vectors = NULL;
}

static FtGpuProfileView P_view(P *p) {
  FtGpuProfileView v;
  memset(&v, 0, sizeof(v));
  v.owner = &p->profile;
  v.serial = p->profile.ftSerial;
  v.weights = p->weights;
  v.codes = p->codes;
  v.vectors = p->vectors;
  v.nVectors = p->nVectors;
  v.codeDist = NULL;
  return v;
}

/* -------------------------------------------------------------- rate model */

static rates_t *make_rates(int nPos, int nRateCats, int mode) {
  int i;
  rates_t *r = (rates_t *)malloc(sizeof(rates_t));
  numeric_t *base = MLSiteRates(nRateCats);
  r->nRateCategories = nRateCats;
  r->rates = (numeric_t *)malloc(sizeof(numeric_t) * (size_t)nRateCats);
  r->ratecat = (unsigned int *)malloc(sizeof(unsigned int) * (size_t)nPos);
  for (i = 0; i < nRateCats; i++) r->rates[i] = base[i];
  base = myfree(base, sizeof(numeric_t) * nRateCats);
  for (i = 0; i < nPos; i++) {
    if (mode == 0)
      r->ratecat[i] = (unsigned int)(i % nRateCats);
    else
      r->ratecat[i] = (unsigned int)((int)(rnd01() * nRateCats) % nRateCats);
  }
  return r;
}

static void free_rates(rates_t *r) {
  free(r->rates);
  free(r->ratecat);
  free(r);
}

/* --------------------------------------------------- PairLogLk comparison */

/* Returns 1 if the case ran on the device, 0 if it was skipped. */
static int compare_pair_loglk(const char *name, P *A, P *B, int nPos, int nCodesLocal,
                              transition_matrix_t *tm, rates_t *rates, double length,
                              int withNonUnitSite) {
  char what[256];
  double *cpuSite = (double *)malloc(sizeof(double) * (size_t)nPos);
  double *init = (double *)malloc(sizeof(double) * (size_t)nPos);
  int i;
  for (i = 0; i < nPos; i++) {
    init[i] = withNonUnitSite ? (0.25 + rnd01()) : 1.0;
    cpuSite[i] = init[i];
  }
  double cpuLk = PairLogLk(&A->profile, &B->profile, length, nPos, tm, rates, cpuSite);

  if (g_ctx == NULL) {
    case_skip("no CUDA device");
    free(cpuSite);
    free(init);
    return 0;
  }

  FtGpuPair pair;
  memset(&pair, 0, sizeof(pair));
  pair.A = P_view(A);
  pair.B = P_view(B);
  pair.length = length;
  pair.site_likelihoods = cpuSite; /* only used by the CPU path */

  FtGpuPairBatch batch;
  memset(&batch, 0, sizeof(batch));
  batch.nPos = nPos;
  batch.model.nCodes = nCodesLocal;
  batch.model.useMatrix = (tm != NULL);
  batch.model.eigenval = (tm != NULL) ? tm->eigenval : NULL;
  batch.model.codeFreq = (tm != NULL) ? &tm->codeFreq[0][0] : NULL;
  batch.rates.nRateCategories = rates->nRateCategories;
  batch.rates.rates = rates->rates;
  batch.rates.ratecat = rates->ratecat;
  batch.nPairs = 1;
  batch.pairs = &pair;
  double gpuLk = 0.0;
  double *gpuSite = (double *)malloc(sizeof(double) * (size_t)nPos);
  batch.out_loglk = &gpuLk;
  batch.out_site = gpuSite;

  FtGpuStatus st = ft_gpu_pair_loglk_batch(g_ctx, &batch);
  if (st != FT_GPU_OK) {
    case_fail("%s: backend error: %s", name, ft_gpu_status_string(st));
  } else {
    snprintf(what, sizeof(what), "%s scalar (len=%.3g)", name, length);
    check_scalar(what, gpuLk, cpuLk);
    /* The host applies site_likelihoods[i] *= lkAB; reproduce that here and
       compare against what the CPU produced from the same initial values. */
    for (i = 0; i < nPos; i++) gpuSite[i] *= init[i];
    snprintf(what, sizeof(what), "%s site", name);
    check_site(what, gpuSite, cpuSite, nPos);
  }
  /* New ordered path must pass a bitwise check, not the old tolerance. */
  int nr=rates->nRateCategories;
  double *tables;
  if(tm)tables=ExpEigenRates(length,tm,rates);
  else {
    tables=malloc((size_t)nr*2*sizeof(double));
    double *ps=PSameVector(length,rates),*pd=PDiffVector(ps,rates);
    memcpy(tables,ps,nr*sizeof(double));memcpy(tables+nr,pd,nr*sizeof(double));
    myfree(ps,nr*sizeof(double));myfree(pd,nr*sizeof(double));
  }
  FtGpuOrderedPairBatch ordered={batch,tables};
  st=ft_gpu_pair_loglk_ordered(g_ctx,&ordered);
  if(st!=FT_GPU_OK)case_fail("ordered status %d",st);
  else {
    if(memcmp(&gpuLk,&cpuLk,sizeof(double)))case_fail("ordered scalar not bitwise equal: %.17g vs %.17g",gpuLk,cpuLk);
    for(i=0;i<nPos;i++)gpuSite[i]*=init[i];
    if(memcmp(gpuSite,cpuSite,(size_t)nPos*sizeof(double)))case_fail("ordered site output not bitwise equal");
  }
  double save=tables[0];tables[0]=NAN;gpuLk=12345;
  if(ft_gpu_pair_loglk_ordered(g_ctx,&ordered)!=FT_GPU_ERR_INVALID||gpuLk!=12345)case_fail("ordered invalid transition contract");
  tables[0]=save;
  if(tm)myfree(tables,(size_t)nr*nCodesLocal*sizeof(double));else free(tables);
  free(cpuSite);
  free(init);
  free(gpuSite);
  return 1;
}

/* ------------------------------------------------------------ dist compare */

static int compare_profile_dist(const char *name, P *A, P *B, int nPos, int nCodesLocal,
                                distance_matrix_t *dmat) {
  char what[256];
  besthit_t cpuHit;
  memset(&cpuHit, 0, sizeof(cpuHit));
  ProfileDist(&A->profile, &B->profile, nPos, dmat, &cpuHit);
  if (g_ctx == NULL) {
    case_skip("no CUDA device");
    return 0;
  }
  FtGpuDistPair pair;
  memset(&pair, 0, sizeof(pair));
  pair.isSeq = 0;
  pair.A = P_view(A);
  pair.B = P_view(B);
  FtGpuDistBatch batch;
  memset(&batch, 0, sizeof(batch));
  batch.nPos = nPos;
  batch.nCodes = nCodesLocal;
  batch.useMatrix = (dmat != NULL);
  batch.dmat_distances = (dmat != NULL) ? &dmat->distances[0][0] : NULL;
  batch.dmat_eigenval = (dmat != NULL) ? dmat->eigenval : NULL;
  batch.dmat_codeFreq = (dmat != NULL) ? &dmat->codeFreq[0][0] : NULL;
  batch.nPairs = 1;
  batch.pairs = &pair;
  double dist = 0.0, weight = 0.0;
  batch.out_dist = &dist;
  batch.out_weight = &weight;
  FtGpuStatus st = ft_gpu_dist_batch(g_ctx, &batch);
  if (st != FT_GPU_OK) {
    case_fail("%s: backend error: %s", name, ft_gpu_status_string(st));
    return 1;
  }
  snprintf(what, sizeof(what), "%s dist", name);
  check_scalar(what, dist, cpuHit.dist);
  snprintf(what, sizeof(what), "%s weight", name);
  check_scalar(what, weight, cpuHit.weight);
  st=ft_gpu_dist_ordered(g_ctx,&batch);
  if(st!=FT_GPU_OK||memcmp(&dist,&cpuHit.dist,sizeof(double))||memcmp(&weight,&cpuHit.weight,sizeof(double)))case_fail("ordered distance not bitwise equal");
  g_nDistChecks++;
  return 1;
}

static int compare_seq_dist(const char *name, const unsigned char *ca, const unsigned char *cb,
                            int nPos, distance_matrix_t *dmat, int nCodesLocal) {
  char what[256];
  besthit_t cpuHit;
  memset(&cpuHit, 0, sizeof(cpuHit));
  SeqDist((unsigned char *)ca, (unsigned char *)cb, nPos, dmat, &cpuHit);
  if (g_ctx == NULL) {
    case_skip("no CUDA device");
    return 0;
  }
  FtGpuDistPair pair;
  memset(&pair, 0, sizeof(pair));
  pair.isSeq = 1;
  pair.codesA = ca;
  pair.codesB = cb;
  FtGpuDistBatch batch;
  memset(&batch, 0, sizeof(batch));
  batch.nPos = nPos;
  batch.nCodes = nCodesLocal;
  batch.useMatrix = (dmat != NULL);
  batch.dmat_distances = (dmat != NULL) ? &dmat->distances[0][0] : NULL;
  batch.dmat_eigenval = (dmat != NULL) ? dmat->eigenval : NULL;
  batch.dmat_codeFreq = (dmat != NULL) ? &dmat->codeFreq[0][0] : NULL;
  batch.nPairs = 1;
  batch.pairs = &pair;
  double dist = 0.0, weight = 0.0;
  batch.out_dist = &dist;
  batch.out_weight = &weight;
  FtGpuStatus st = ft_gpu_dist_batch(g_ctx, &batch);
  if (st != FT_GPU_OK) {
    case_fail("%s: backend error: %s", name, ft_gpu_status_string(st));
    return 1;
  }
  snprintf(what, sizeof(what), "%s dist", name);
  check_scalar(what, dist, cpuHit.dist);
  snprintf(what, sizeof(what), "%s weight", name);
  check_scalar(what, weight, cpuHit.weight);
  g_nDistChecks++;
  return 1;
}

/* ------------------------------------------------------- test 1: the index */

static void test_vector_index(void) {
  int styles[4] = {STYLE_CODES, STYLE_VECTORS, STYLE_MIXED, STYLE_GAPPY};
  int s;
  case_begin("vector_index/GET_FREQ agreement");
  for (s = 0; s < 4; s++) {
    int nPos = 257; /* deliberately not a multiple of any block size */
    P p;
    P_build2(&p, nPos, 20, styles[s], NULL, 0);
    int *index = (int *)malloc(sizeof(int) * (size_t)nPos);
    int nVec = -1;
    if (ft_pack_vector_index(p.weights, p.codes, nPos, index, &nVec) != 0) {
      case_fail("style %d: ft_pack_vector_index failed", s);
    } else {
      int iFreq = 0;
      int bad = 0;
      int i;
      for (i = 0; i < nPos; i++) {
        int expect = -1;
        if (p.weights[i] > 0 && p.codes[i] == NOCODE) {
          expect = iFreq;
          iFreq++;
        }
        if (index[i] != expect) bad++;
      }
      if (bad) case_fail("style %d: %d index entries disagree with GET_FREQ", s, bad);
      if (nVec != p.nVectors)
        case_fail("style %d: nVectors %d != profile nVectors %d", s, nVec, p.nVectors);
      if (iFreq != p.nVectors)
        case_fail("style %d: GET_FREQ walk counted %d != %d", s, iFreq, p.nVectors);
    }
    free(index);
    P_free(&p);
  }
  case_pass();
}

/* ------------------------------------------- test 2: synthetic PairLogLk */

static void test_pair_loglk_protein(void) {
  int nPos = 129; /* not a multiple of 32 or 256 */
  int nRateCats = 20;
  rates_t *rates = make_rates(nPos, nRateCats, 1);
  rates_t *ratesRR = make_rates(nPos, 20, 0);
  transition_matrix_t *models[3];
  const char *modelNames[3] = {"JTT", "LG", "WAG"};
  int m, style, eigen, li;
  double lengths[5] = {1e-12, 2.5e-9, 0.05, 1.0, 6.0};
  models[0] = CreateTransitionMatrix(matrixJTT92, statJTT92);
  models[1] = CreateTransitionMatrix(matrixLG08, statLG08);
  models[2] = CreateTransitionMatrix(matrixWAG01, statWAG01);
  for (m = 0; m < 3; m++) {
    for (style = 0; style < 4; style++) {
      for (eigen = 0; eigen < 2; eigen++) {
        P A, B;
        char name[192];
        P_build2(&A, nPos, 20, style, models[m]->codeFreq, 1);
        P_build2(&B, nPos, 20, style, models[m]->codeFreq, 1);
        for (li = 0; li < 5; li++) {
          snprintf(name, sizeof(name), "pairloglk/protein/%s/style%d/vec%d/len%.3g",
                   modelNames[m], style, eigen, lengths[li]);
          case_begin(name);
          if (compare_pair_loglk(name, &A, &B, nPos, 20, models[m], rates, lengths[li], 0))
            case_pass();
        }
        P_free(&A);
        P_free(&B);
      }
    }
  }
  /* round-robin rate categories and a batch of many pairs */
  {
    P A, B;
    char name[192];
    P_build2(&A, nPos, 20, STYLE_MIXED, models[0]->codeFreq, 1);
    P_build2(&B, nPos, 20, STYLE_MIXED, models[0]->codeFreq, 1);
    snprintf(name, sizeof(name), "pairloglk/protein/roundrobin-rates");
    case_begin(name);
    if (compare_pair_loglk(name, &A, &B, nPos, 20, models[0], ratesRR, 0.3, 1)) case_pass();
    P_free(&A);
    P_free(&B);
  }
  if (g_ctx != NULL) {
    enum { NB = 37 };
    P *As = (P *)malloc(sizeof(P) * NB);
    P *Bs = (P *)malloc(sizeof(P) * NB);
    FtGpuPair *pairs = (FtGpuPair *)malloc(sizeof(FtGpuPair) * NB);
    double *gpuLk = (double *)malloc(sizeof(double) * NB);
    double *cpuLk = (double *)malloc(sizeof(double) * NB);
    int i;
    case_begin("pairloglk/batch37");
    for (i = 0; i < NB; i++) {
      P_build2(&As[i], nPos, 20, i % 4, models[0]->codeFreq, 1);
      P_build2(&Bs[i], nPos, 20, (i + 1) % 4, models[0]->codeFreq, 1);
      memset(&pairs[i], 0, sizeof(FtGpuPair));
      pairs[i].A = P_view(&As[i]);
      pairs[i].B = P_view(&Bs[i]);
      pairs[i].length = 1e-9 * (i + 1) * 0.37;
      cpuLk[i] = PairLogLk(&As[i].profile, &Bs[i].profile, pairs[i].length, nPos, models[0],
                           rates, NULL);
    }
    FtGpuPairBatch batch;
    memset(&batch, 0, sizeof(batch));
    batch.nPos = nPos;
    batch.model.nCodes = 20;
    batch.model.useMatrix = 1;
    batch.model.eigenval = models[0]->eigenval;
    batch.model.codeFreq = &models[0]->codeFreq[0][0];
    batch.rates.nRateCategories = rates->nRateCategories;
    batch.rates.rates = rates->rates;
    batch.rates.ratecat = rates->ratecat;
    batch.nPairs = NB;
    batch.pairs = pairs;
    batch.out_loglk = gpuLk;
    batch.out_site = NULL;
    FtGpuStatus st = ft_gpu_pair_loglk_batch(g_ctx, &batch);
    if (st != FT_GPU_OK) {
      case_fail("batch37: backend error: %s", ft_gpu_status_string(st));
    } else {
      for (i = 0; i < NB; i++) check_scalar("batch37", gpuLk[i], cpuLk[i]);
    }
    batch.nPairs = 0;
    if (ft_gpu_pair_loglk_batch(g_ctx, &batch) != FT_GPU_OK)
      case_fail("empty batch was rejected");
    case_pass();
    for (i = 0; i < NB; i++) {
      P_free(&As[i]);
      P_free(&Bs[i]);
    }
    free(As);
    free(Bs);
    free(pairs);
    free(gpuLk);
    free(cpuLk);
  } else {
    case_begin("pairloglk/batch37");
    case_skip("no CUDA device");
  }
  /* L == 1 */
  {
    P A, B;
    char name[64];
    P_build2(&A, 1, 20, STYLE_MIXED, models[0]->codeFreq, 1);
    P_build2(&B, 1, 20, STYLE_MIXED, models[0]->codeFreq, 1);
    snprintf(name, sizeof(name), "pairloglk/protein/L1");
    case_begin(name);
    if (compare_pair_loglk(name, &A, &B, 1, 20, models[0], rates, 0.1, 0)) case_pass();
    P_free(&A);
    P_free(&B);
  }
  for (m = 0; m < 3; m++) free(models[m]);
  free_rates(rates);
  free_rates(ratesRR);
}

static void test_pair_loglk_dna(void) {
  int nPos = 129;
  rates_t *rates = make_rates(nPos, 8, 1);
  double gtrrates[6] = {1.0, 2.0, 1.0, 1.0, 2.0, 1.0};
  double gtrfreq[4] = {0.3, 0.2, 0.2, 0.3};
  transition_matrix_t *tmGtr = CreateGTR(gtrrates, gtrfreq);
  int mi, style, eigen, li;
  double lengths[5] = {1e-12, 2.5e-9, 0.05, 1.0, 6.0};
  for (mi = 0; mi < 2; mi++) {
    for (style = 0; style < 4; style++) {
      for (eigen = 0; eigen < 2; eigen++) {
        P A, B;
        char name[192];
        P_build2(&A, nPos, 4, style, mi == 0 ? NULL : tmGtr->codeFreq, mi == 0 ? 0 : 1);
        P_build2(&B, nPos, 4, style, mi == 0 ? NULL : tmGtr->codeFreq, mi == 0 ? 0 : 1);
        for (li = 0; li < 5; li++) {
          snprintf(name, sizeof(name), "pairloglk/DNA/%s/style%d/eigen%d/len%.3g",
                   mi == 0 ? "JC" : "GTR", style, eigen, lengths[li]);
          case_begin(name);
          if (compare_pair_loglk(name, &A, &B, nPos, 4, mi == 0 ? NULL : tmGtr, rates,
                                 lengths[li], 0))
            case_pass();
        }
        P_free(&A);
        P_free(&B);
      }
    }
  }
  free(tmGtr);
  free_rates(rates);
}

/* ------------------------------------------------ test 3: distance kernels */

/* A structurally valid distance_matrix_t: distances are the spectral
   decomposition of a random symmetric matrix, so SetupDistanceMatrix()'s
   consistency check passes and codeFreq rows are genuine eigenspace vectors. */
static void make_test_dmat(distance_matrix_t *dmat, int nCodesLocal) {
  double a[MAXCODES * MAXCODES], eval[MAXCODES], e[MAXCODES];
  int i, j, k;
  double np = (double)nCodesLocal;
  memset(dmat, 0, sizeof(*dmat));
  for (i = 0; i < nCodesLocal; i++) {
    dmat->distances[i][i] = 0.0;
    for (j = i + 1; j < nCodesLocal; j++) {
      double d = 0.05 + 0.9 * ((i * 7 + j * 3 + i * j) % 11) / np;
      dmat->distances[i][j] = d;
      dmat->distances[j][i] = d;
    }
  }
  for (i = 0; i < nCodesLocal; i++)
    for (j = 0; j < nCodesLocal; j++) a[nCodesLocal * i + j] = dmat->distances[i][j];
  tred2(a, nCodesLocal, nCodesLocal, eval, e);
  tqli(eval, e, nCodesLocal, nCodesLocal, a);
  for (k = 0; k < nCodesLocal; k++) dmat->eigenval[k] = eval[k];
  /* FastTree's tqli() returns eigenvectors in rows: z[k][i] is component i of
     eigenvector k, and distances[i][j] = sum_k eigenval[k]*z[k][i]*z[k][j]. */
  for (k = 0; k < nCodesLocal; k++)
    for (i = 0; i < nCodesLocal; i++) dmat->eigeninv[k][i] = a[nCodesLocal * k + i];
  SetupDistanceMatrix(dmat); /* validates and fills codeFreq/eigentot/gapFreq */
}

static void test_dist_synthetic(void) {
  int nPos = 333;
  int style, eigen;
  distance_matrix_t dmat;
  case_begin("dist/synthetic");
  make_test_dmat(&dmat, 20);
  for (style = 0; style < 4; style++) {
    for (eigen = 0; eigen < 2; eigen++) {
      P A, B;
      char name[160];
      /* eigen == 0: probability-like vectors (no matrix semantics);
         eigen == 1: legitimate eigenspace vectors of this dmat */
      P_build2(&A, nPos, 20, style, eigen ? dmat.codeFreq : NULL, 0);
      P_build2(&B, nPos, 20, style, eigen ? dmat.codeFreq : NULL, 0);
      snprintf(name, sizeof(name), "dist/protein/style%d/vec%d", style, eigen);
      compare_profile_dist(name, &A, &B, nPos, 20, NULL);
      snprintf(name, sizeof(name), "dist/protein/matrix/style%d/vec%d", style, eigen);
      compare_profile_dist(name, &A, &B, nPos, 20, &dmat);
      P_free(&A);
      P_free(&B);
    }
  }
  /* degenerate: no overlapping data at all */
  {
    P A, B, C;
    char name[64];
    P_build_empty(&A, nPos, 20);
    P_build2(&B, nPos, 20, STYLE_MIXED, dmat.codeFreq, 0);
    P_build_empty(&C, nPos, 20);
    snprintf(name, sizeof(name), "dist/nooverlap/profile");
    compare_profile_dist(name, &A, &B, nPos, 20, NULL);
    compare_profile_dist("dist/nooverlap/profile-matrix", &A, &B, nPos, 20, &dmat);
    compare_profile_dist("dist/nooverlap/both-empty", &A, &C, nPos, 20, NULL);
    P_free(&A);
    P_free(&B);
    P_free(&C);
  }
  /* sequence distances, including the no-overlap case */
  {
    unsigned char *ca = (unsigned char *)malloc((size_t)nPos);
    unsigned char *cb = (unsigned char *)malloc((size_t)nPos);
    unsigned char *call = (unsigned char *)malloc((size_t)nPos);
    int i;
    make_test_dmat(&dmat, 20);
    for (i = 0; i < nPos; i++) {
      ca[i] = (rnd01() < 0.2) ? NOCODE : (unsigned char)(rnd01() * 20);
      cb[i] = (rnd01() < 0.2) ? NOCODE : (unsigned char)(rnd01() * 20);
      call[i] = NOCODE;
    }
    compare_seq_dist("dist/seq/dmat", ca, cb, nPos, &dmat, 20);
    compare_seq_dist("dist/seq/nomatrix", ca, cb, nPos, NULL, 20);
    compare_seq_dist("dist/seq/nooverlap", ca, call, nPos, &dmat, 20);
    compare_seq_dist("dist/seq/L1", ca, cb, 1, &dmat, 20);
    free(ca);
    free(cb);
    free(call);
  }
  case_pass();
}

/* ------------------------------------------------------- test 4: real data */

typedef struct {
  char **names;
  char **seqs;
  int n;
  int nPos;
} aln_t;

static aln_t *read_fasta(const char *path) {
  FILE *fp = fopen(path, "r");
  aln_t *a;
  char *cur = NULL;
  int curLen = 0;
  int cap = 16;
  char buf[1 << 16];
  if (fp == NULL) return NULL;
  a = (aln_t *)calloc(1, sizeof(aln_t));
  a->names = (char **)malloc(sizeof(char *) * (size_t)cap);
  a->seqs = (char **)malloc(sizeof(char *) * (size_t)cap);
  cur = (char *)malloc(1 << 20);
  cur[0] = '\0';
  while (fgets(buf, sizeof(buf), fp) != NULL) {
    char *nl = strchr(buf, '\n');
    int len;
    if (nl != NULL) *nl = '\0';
    nl = strchr(buf, '\r');
    if (nl != NULL) *nl = '\0';
    if (buf[0] == '>') {
      if (curLen > 0) {
        if (a->n == cap) {
          cap *= 2;
          a->names = (char **)realloc(a->names, sizeof(char *) * (size_t)cap);
          a->seqs = (char **)realloc(a->seqs, sizeof(char *) * (size_t)cap);
        }
        a->names[a->n] = strdup(buf + 1);
        a->seqs[a->n] = cur;
        a->n++;
        cur = (char *)malloc(1 << 20);
        curLen = 0;
        cur[0] = '\0';
      }
      continue;
    }
    len = (int)strlen(buf);
    if (curLen + len >= (1 << 20) - 1) break;
    memcpy(cur + curLen, buf, (size_t)len);
    curLen += len;
    cur[curLen] = '\0';
  }
  if (curLen > 0) {
    if (a->n == cap) {
      cap += 1;
      a->names = (char **)realloc(a->names, sizeof(char *) * (size_t)cap);
      a->seqs = (char **)realloc(a->seqs, sizeof(char *) * (size_t)cap);
    }
    a->names[a->n] = strdup("last");
    a->seqs[a->n] = cur;
    a->n++;
  } else {
    free(cur);
  }
  fclose(fp);
  if (a->n > 0) a->nPos = (int)strlen(a->seqs[0]);
  return a;
}

static void test_real_alignment(const char *path) {
  aln_t *a = read_fasta(path);
  unsigned long counts[256];
  int i;
  transition_matrix_t *tm;
  rates_t *rates;
  if (a == NULL || a->n < 8 || a->nPos < 2) {
    case_begin("real_alignment");
    case_fail("could not read at least 8 sequences from %s", path ? path : "(none)");
    return;
  }
  memset(counts, 0, sizeof(counts));
  printf("# real alignment %s: %d sequences x %d positions (nCodes=%d)\n", path, a->n, a->nPos,
         nCodes);
  if (nCodes == 20) {
    tm = CreateTransitionMatrix(matrixJTT92, statJTT92);
  } else {
    double gtrrates[6] = {1.0, 2.0, 1.0, 1.0, 2.0, 1.0};
    double gtrfreq[4] = {0.25, 0.25, 0.25, 0.25};
    tm = CreateGTR(gtrrates, gtrfreq);
  }
  rates = make_rates(a->nPos, 20, 1);
  {
    int nUse = a->n < 12 ? a->n : 12;
    profile_t **prof = (profile_t **)malloc(sizeof(profile_t *) * (size_t)nUse);
    for (i = 0; i < nUse; i++)
      prof[i] = SeqToProfile(NULL, a->seqs[i], a->nPos, NULL, 0, i, counts);

    case_begin("real/leaf_index");
    {
      int bad = 0;
      for (i = 0; i < nUse; i++) {
        int nVec = ft_count_vectors(prof[i]->weights, prof[i]->codes, a->nPos);
        if (nVec != prof[i]->nVectors) bad++;
      }
      if (bad) case_fail("%d real leaf profiles disagree on nVectors", bad);
    }
    case_pass();

    for (i = 0; i + 1 < nUse; i += 2) {
      char name[128];
      P A, B;
      snprintf(name, sizeof(name), "real/leafleaf/%d-%d", i, i + 1);
      case_begin(name);
      P_wrap(&A, prof[i]);
      P_wrap(&B, prof[i + 1]);
      if (compare_pair_loglk(name, &A, &B, a->nPos, nCodes, tm, rates, 0.1, 0)) case_pass();
      P_free(&A);
      P_free(&B);
    }
    {
      profile_t *pAB = PosteriorProfile(prof[0], prof[1], 0.07, 0.12, tm, rates, a->nPos, 0);
      profile_t *pCD = PosteriorProfile(prof[2], prof[3], 0.2, 0.03, tm, rates, a->nPos, 0);
      char name[128];
      P A, B;
      case_begin("real/posterior_index");
      {
        int nVec = ft_count_vectors(pAB->weights, pAB->codes, a->nPos);
        if (nVec != pAB->nVectors)
          case_fail("posterior index count %d != nVectors %d", nVec, pAB->nVectors);
        nVec = ft_count_vectors(pCD->weights, pCD->codes, a->nPos);
        if (nVec != pCD->nVectors)
          case_fail("posterior index count %d != nVectors %d", nVec, pCD->nVectors);
      }
      case_pass();

      P_wrap(&A, pAB);
      P_wrap(&B, prof[4]);
      snprintf(name, sizeof(name), "real/posterior-vs-leaf");
      case_begin(name);
      if (compare_pair_loglk(name, &A, &B, a->nPos, nCodes, tm, rates, 0.25, 1)) case_pass();

      P_free(&B);
      P_wrap(&B, pCD);
      snprintf(name, sizeof(name), "real/posterior-vs-posterior");
      case_begin(name);
      if (compare_pair_loglk(name, &A, &B, a->nPos, nCodes, tm, rates, 0.02, 0)) case_pass();

      snprintf(name, sizeof(name), "real/profiledist");
      case_begin(name);
      compare_profile_dist(name, &A, &B, a->nPos, nCodes, NULL);
      case_pass();

      P_free(&A);
      P_free(&B);
      pAB = FreeProfile(pAB, a->nPos, 0);
      pCD = FreeProfile(pCD, a->nPos, 0);
    }
    case_begin("real/seqdist");
    compare_seq_dist("real/seqdist", prof[5]->codes, prof[6]->codes, a->nPos, NULL, nCodes);
    case_pass();
    for (i = 0; i < nUse; i++) prof[i] = FreeProfile(prof[i], a->nPos, 0);
    free(prof);
  }
  free_rates(rates);
  free(tm);
  for (i = 0; i < a->n; i++) {
    free(a->names[i]);
    free(a->seqs[i]);
  }
  free(a->names);
  free(a->seqs);
  free(a);
}

/* ------------------------------------------------------- fallback routing */

static void test_fallback_policy(void) {
  case_begin("fallback/thresholds");
  {
    P A, B;
    transition_matrix_t *tm;
    rates_t *rates;
    FtGpuPair pair;
    double lk = 0.0;
    int r;
    tm = CreateTransitionMatrix(matrixJTT92, statJTT92);
    P_build2(&A, 64, 20, STYLE_MIXED, tm->codeFreq, 1);
    P_build2(&B, 64, 20, STYLE_MIXED, tm->codeFreq, 1);
    rates = make_rates(64, 4, 0);
    memset(&pair, 0, sizeof(pair));
    pair.A = P_view(&A);
    pair.B = P_view(&B);
    pair.length = 0.1;
    ftb_configure("auto", -1, 0, 0, 1, 100000000L, 1);
    r = ftb_pair_loglk_batch(64, 20, 1, tm->eigenval, &tm->codeFreq[0][0],
                             rates->nRateCategories, rates->rates, rates->ratecat, 1, &pair,
                             &lk, NULL);
    if (r != FTB_FALLBACK) case_fail("expected a threshold fallback, got %d", r);
    ftb_configure("cpu", -1, 0, 0, 0, 0, 1);
    if (ftb_active()) case_fail("--backend cpu must not activate the device");
    free_rates(rates);
    free(tm);
    P_free(&A);
    P_free(&B);
  }
  case_pass();
}


/* Error-path regression tests: outputs are committed only on success. */
static void test_api_errors(void) {
  P A, B;
  rates_t *rates;
  FtGpuPair pair = {0};
  FtGpuPairBatch batch = {0};
  double lk = 12345, site[64], expected[64];
  FtGpuStatus st;
  int before = g_fail;
  case_begin("api/errors-and-site-output");
  if (!g_ctx) { case_skip("no CUDA device"); return; }
  nCodes = 4;
  P_build2(&A, 64, 4, STYLE_MIXED, NULL, 0);
  P_build2(&B, 64, 4, STYLE_MIXED, NULL, 0);
  rates = make_rates(64, 4, 0);
  pair.A = P_view(&A); pair.B = P_view(&B); pair.length = 0.1;
  batch.nPos = 64; batch.model.nCodes = 4;
  batch.rates.nRateCategories = 4; batch.rates.rates = rates->rates;
  batch.rates.ratecat = rates->ratecat;
  batch.nPairs = 1; batch.pairs = &pair; batch.out_loglk = &lk; batch.out_site = site;
  for (int i=0;i<64;i++) site[i]=12345, expected[i]=1;
  double ref = PairLogLk(&A.profile, &B.profile, 0.1, 64, NULL, rates, expected);
  st = ft_gpu_pair_loglk_batch(g_ctx, &batch);
  if (st != FT_GPU_OK) case_fail("out_site without marker returned %d", st);
  else { check_scalar("scalar", lk, ref); check_site("out_site", site, expected, 64); }
  lk=12345; site[0]=12345;
  rates->ratecat[0]=4;
  st=ft_gpu_pair_loglk_batch(g_ctx,&batch);
  if (st!=FT_GPU_ERR_INVALID || lk!=12345 || site[0]!=12345)
    case_fail("invalid ratecat modified outputs or status=%d",st);
  rates->ratecat[0]=0;
  pair.A.nVectors++;
  st=ft_gpu_pair_loglk_batch(g_ctx,&batch);
  if (st!=FT_GPU_ERR_INVALID || lk!=12345) case_fail("cache shape check status=%d",st);
  pair.A.nVectors--;
  pair.length=NAN;
  if (ft_gpu_pair_loglk_batch(g_ctx,&batch)!=FT_GPU_ERR_INVALID) case_fail("NaN length accepted");
  pair.length=0.1;
  batch.nPos=0;
  if (ft_gpu_pair_loglk_batch(g_ctx,&batch)!=FT_GPU_ERR_INVALID) case_fail("zero sites accepted");
  batch.nPos=64;
  if (ft_gpu_pair_loglk_batch(g_ctx,&batch)!=FT_GPU_OK) case_fail("context did not recover");
  /* A zero model creates lkAB=0: reject atomically rather than rescale forever. */
  double eigen[4]={0}, freq[FT_GPU_CODE_ROWS*FT_GPU_CODE_STRIDE]={0};
  batch.model.useMatrix=1; batch.model.eigenval=eigen; batch.model.codeFreq=freq;
  lk=12345; for(int i=0;i<64;i++) site[i]=12345;
  st=ft_gpu_pair_loglk_batch(g_ctx,&batch);
  if(st!=FT_GPU_ERR_INTERNAL || lk!=12345) case_fail("zero likelihood status=%d",st);
  for(int i=0;i<64;i++) if(site[i]!=12345) { case_fail("failed output partially committed"); break; }
  batch.model.useMatrix=0;
  if(ft_gpu_pair_loglk_batch(g_ctx,&batch)!=FT_GPU_OK) case_fail("numerical failure poisoned context");
  if(g_fail==before) case_pass();

  case_begin("api/recycled-storage-new-content"); before=g_fail;
  FtGpuConfig reuseCfg={0}; FtGpuContext *reuse=NULL;
  reuseCfg.device_id=-1;
  if(ft_gpu_create(&reuseCfg,&reuse)!=FT_GPU_OK) case_fail("reuse context creation failed");
  else {
    FtGpuStats oldStats, newStats;
    if(ft_gpu_pair_loglk_batch(reuse,&batch)!=FT_GPU_OK) case_fail("initial reuse batch failed");
    ft_gpu_stats_get(reuse,&oldStats);
    ft_gpu_profile_invalidate(reuse,pair.A.owner,pair.A.serial);
    for(int i=0;i<64;i++) A.weights[i]*=0.5;
    pair.A.serial++;
    for(int i=0;i<64;i++) expected[i]=1;
    ref=PairLogLk(&A.profile,&B.profile,0.1,64,NULL,rates,expected);
    if(ft_gpu_pair_loglk_batch(reuse,&batch)!=FT_GPU_OK) case_fail("recycled storage batch failed");
    else { check_scalar("recycled scalar",lk,ref); check_site("recycled sites",site,expected,64); }
    ft_gpu_stats_get(reuse,&newStats);
    if(newStats.nProfileUploads!=oldStats.nProfileUploads+1 || newStats.deviceBytes!=oldStats.deviceBytes)
      case_fail("profile invalidation did not reuse storage and upload fresh content");
    ft_gpu_destroy(reuse);
    for(int i=0;i<64;i++) A.weights[i]*=2.0;
    pair.A.serial--;
  }
  if(g_fail==before) case_pass();

  case_begin("api/pinned-budget-and-accounting"); before=g_fail;
  FtGpuConfig cfg={0}; FtGpuContext *small=NULL;
  cfg.device_id=-1;
  /* Each profile exceeds 64*(8+1+4); budget fits one but not both. */
  cfg.memory_limit_bytes=64*13+(size_t)A.nVectors*4*8;
  if(ft_gpu_create(&cfg,&small)!=FT_GPU_OK) case_fail("small context creation failed");
  else {
    lk=12345; site[0]=12345;
    st=ft_gpu_pair_loglk_batch(small,&batch);
    FtGpuStats stats; ft_gpu_stats_get(small,&stats);
    if(st!=FT_GPU_ERR_MEMORY || lk!=12345 || site[0]!=12345)
      case_fail("budget failure status=%d outputs=%g/%g",st,lk,site[0]);
    if(stats.deviceBytes>cfg.memory_limit_bytes || stats.peakDeviceBytes>cfg.memory_limit_bytes)
      case_fail("budget exceeded or accounting underflow");
    ft_gpu_profile_invalidate(small,pair.A.owner,pair.A.serial);
    ft_gpu_stats_get(small,&stats);
    if(stats.deviceBytes!=0) case_fail("failed batch leaked accounted bytes");
    ft_gpu_destroy(small);
  }
  if(g_fail==before) case_pass();

  case_begin("api/scratch-budget"); before=g_fail;
  cfg.memory_limit_bytes=2*64*13+(size_t)(A.nVectors+B.nVectors)*4*8;
  small=NULL;
  if(ft_gpu_create(&cfg,&small)!=FT_GPU_OK) case_fail("scratch context creation failed");
  else {
    lk=12345;
    st=ft_gpu_pair_loglk_batch(small,&batch);
    FtGpuStats stats; ft_gpu_stats_get(small,&stats);
    if(st!=FT_GPU_ERR_MEMORY || lk!=12345 || stats.deviceBytes>cfg.memory_limit_bytes)
      case_fail("scratch ignored budget or modified output (status=%d)",st);
    ft_gpu_destroy(small);
  }
  if(g_fail==before) case_pass();

  case_begin("api/distance-invalid-codes"); before=g_fail;
  unsigned char ca[64]={0}, cb[64]={0}; double dist=12345, weight=12345;
  FtGpuDistPair dp={0}; FtGpuDistBatch db={0};
  dp.isSeq=1; dp.codesA=ca; dp.codesB=cb; cb[63]=4;
  db.nPos=64; db.nCodes=4; db.nPairs=1; db.pairs=&dp;
  db.out_dist=&dist; db.out_weight=&weight;
  st=ft_gpu_dist_batch(g_ctx,&db);
  if(st!=FT_GPU_ERR_INVALID || dist!=12345 || weight!=12345)
    case_fail("invalid sequence code status=%d",st);
  cb[63]=0;
  if(ft_gpu_dist_batch(g_ctx,&db)!=FT_GPU_OK) case_fail("distance context did not recover");
  if(g_fail==before) case_pass();

  case_begin("fallback/unsupported-rate-count"); before=g_fail;
  ftb_configure("auto",-1,0,0,0,0,1);
  if(ftb_pair_loglk_batch(64,4,0,NULL,NULL,65,rates->rates,rates->ratecat,1,&pair,&lk,NULL)
      !=FTB_FALLBACK) case_fail("65 categories did not fall back");
  ftb_configure("cpu",-1,0,0,0,0,1);
  if(g_fail==before) case_pass();
  ft_gpu_profile_invalidate(g_ctx,pair.A.owner,pair.A.serial);
  ft_gpu_profile_invalidate(g_ctx,pair.B.owner,pair.B.serial);
  free_rates(rates); P_free(&A); P_free(&B);
}

/* ------------------------------------------------------------------- main */

static void test_resident_matrix(int nc) {
  bool savedExact=exactML;
  const int ns=5,np=129,nj=4;
  int children[8]={0,1,2,3,5,6,7,4};
  double lengths[4][2]={{0,0.02},{0.5,0.001},{0.03,0.04},{0,0.08}};
  int categories[]={1,4,16,64};
  for(int c=0;c<4;c++) {
    char name[80];snprintf(name,sizeof(name),"resident/matrix-%d-rates-%d",nc,categories[c]);case_begin(name);
    if(!g_ctx) {case_skip("no CUDA device");continue;}
    int before=g_fail;nCodes=nc;
    P leaf[5];unsigned char codes[5*129];
    for(int l=0;l<ns;l++) {
      P_build2(&leaf[l],np,nc,STYLE_GAPPY,NULL,0);
      leaf[l].codes[0]=NOCODE;leaf[l].weights[0]=0;
      memcpy(codes+l*np,leaf[l].codes,np);
    }
    rates_t *rates=make_rates(np,categories[c],0);
    if(categories[c]==1)rates->rates[0]=1.0; /* production -nocat convention */
    FtGpuResidentTree *tree=NULL;
    FtGpuStats initial,after;ft_gpu_stats_get(g_ctx,&initial);
    FtGpuStatus st=ft_gpu_ml_tree_create(g_ctx,ns,np,nj,nc,categories[c],codes,rates->ratecat,children,&tree);
    if(st!=FT_GPU_OK)case_fail("create status %d",st);
    double *exps=malloc((size_t)nj*3*categories[c]*nc*sizeof(double));
    double model[1643];
    if(tree)for(int trial=0;trial<4;trial++) {
      double gr[6]={1,2,1,1,2,1};gr[trial]=0.1+trial*2.0;
      double freq[4]={0.3,0.2,0.2,0.3};
      transition_matrix_t *tm;
      if(nc==4)tm=CreateGTR(gr,freq);
      else if(trial==1)tm=CreateTransitionMatrix(matrixLG08,statLG08);
      else if(trial==2)tm=CreateTransitionMatrix(matrixWAG01,statWAG01);
      else tm=CreateTransitionMatrix(matrixJTT92,statJTT92);
      exactML=nc==20 && trial==3;
      int stat=(nc+1)*nc,inv=stat+nc;
      for(int j=0;j<=nc;j++)for(int k=0;k<nc;k++)model[j*nc+k]=tm->codeFreq[j==nc?NOCODE:j][k];
      for(int j=0;j<nc;j++) {model[stat+j]=tm->statinv[j];for(int k=0;k<nc;k++)model[inv+j*nc+k]=nc==4?tm->eigeninvT[j][k]:tm->eigeninv[j][k];}
      if(nc==20) {
        for(int j=0;j<20;j++)for(int k=0;k<20;k++) {
          model[inv+400+j*20+k]=tm->nearP[j][k];model[inv+800+j*20+k]=tm->nearFreq[j][k];
        }
        model[1640]=exactML?1:0;model[1641]=approxMLminf;model[1642]=approxMLminratio;
      }
      profile_t *profiles[9];for(int l=0;l<ns;l++)profiles[l]=&leaf[l].profile;
      double reference=0,siteCPU[129],siteCorrection[129]={0},siteGPU[129];
      for(int i=0;i<np;i++)siteCPU[i]=1.0;
      for(int j=0;j<nj;j++) {
        for(int part=0;part<3;part++) {
          double length=part==2?lengths[j][0]+lengths[j][1]:lengths[j][part];
          if(part<2&&length<MLMinBranchLength)length=MLMinBranchLength;
          numeric_t *e=ExpEigenRates(length,tm,rates);
          memcpy(exps+(size_t)(j*3+part)*categories[c]*nc,e,(size_t)categories[c]*nc*sizeof(double));
          myfree(e,(size_t)categories[c]*nc*sizeof(double));
        }
        if(g_verbose)fprintf(stderr,"resident nr=%d trial=%d job=%d\n",categories[c],trial,j);
      reference+=PairLogLk(profiles[children[2*j]],profiles[children[2*j+1]],lengths[j][0]+lengths[j][1],np,tm,rates,siteCPU);
        if(j<nj-1)for(int i=0;i<np;i++)while(siteCPU[i]<LkUnderflow) {siteCPU[i]*=LkUnderflowInv;siteCorrection[i]-=LogLkUnderflow;}
        if(j<nj-1)profiles[ns+j]=PosteriorProfile(profiles[children[2*j]],profiles[children[2*j+1]],lengths[j][0],lengths[j][1],tm,rates,np,0);
      }
      double value=12345;
      st=ft_gpu_ml_tree_eval(tree,exps,model,&value,siteGPU);
      for(int i=0;i<np;i++)siteCPU[i]=siteCorrection[i]+log(siteCPU[i]);
      if(st==FT_GPU_OK&&memcmp(siteCPU,siteGPU,sizeof(siteCPU)))case_fail("resident site output not bitwise equal");
      if(st!=FT_GPU_OK||value!=reference)case_fail("trial %d status %d exact mismatch %.17g vs %.17g",trial,st,value,reference);
      for(int j=0;j<nj-1;j++)FreeProfile(profiles[ns+j],np,0);
      myfree(tm,sizeof(*tm));
    }
    if(tree) {
      double value=12345,old=exps[0];exps[0]=NAN;
      if(ft_gpu_ml_tree_eval(tree,exps,model,&value,NULL)!=FT_GPU_ERR_INVALID||value!=12345)case_fail("invalid input output contract");
      exps[0]=old;double bad[1643]={0};
      if(ft_gpu_ml_tree_eval(tree,exps,bad,&value,NULL)!=FT_GPU_ERR_INTERNAL||value!=12345)case_fail("invalid numerical output contract");
      if(ft_gpu_ml_tree_eval(tree,exps,model,&value,NULL)!=FT_GPU_OK)case_fail("resident failed to recover");
    }
    ft_gpu_ml_tree_destroy(tree);ft_gpu_stats_get(g_ctx,&after);
    if(after.deviceBytes!=initial.deviceBytes)case_fail("resident storage leaked");
    FtGpuResidentTree *badTree=NULL;int oldChild=children[0];children[0]=ns;
    if(ft_gpu_ml_tree_create(g_ctx,ns,np,nj,nc,categories[c],codes,rates->ratecat,children,&badTree)!=FT_GPU_ERR_INVALID||badTree)case_fail("cyclic topology accepted");
    children[0]=oldChild;
    FtGpuConfig cfg={0};cfg.device_id=-1;cfg.memory_limit_bytes=(size_t)(ns+nj)*np;FtGpuContext *small=NULL; /* leaf+gap buffers fit; vectors must fail */
    if(ft_gpu_create(&cfg,&small)!=FT_GPU_OK)case_fail("small context failed");
    else {
      if(ft_gpu_ml_tree_create(small,ns,np,nj,nc,categories[c],codes,rates->ratecat,children,&badTree)!=FT_GPU_ERR_MEMORY||badTree)case_fail("resident budget ignored");
      ft_gpu_stats_get(small,&after);if(after.deviceBytes!=0||after.peakDeviceBytes>cfg.memory_limit_bytes)case_fail("failed create leaked storage or exceeded budget");ft_gpu_destroy(small);
    }
    free(exps);free_rates(rates);for(int l=0;l<ns;l++)P_free(&leaf[l]);
    if(g_fail==before)case_pass();
  }
  exactML=savedExact;
}


static void test_jc_posterior(void) {
  nCodes=4;codesString=codesStringNT;
  const int counts[3]={1,4,64};
  for(int rc=0;rc<3;rc++)for(int style=0;style<4;style++)for(int trial=0;trial<2;trial++) {
    char name[100];snprintf(name,sizeof(name),"JC posterior exact rates=%d style=%d trial=%d",counts[rc],style,trial);case_begin(name);
    if(!g_ctx){case_skip("no GPU");continue;}
    int before=g_fail,np=257,nr=counts[rc];
    P a,b;P_build2(&a,np,4,style,NULL,0);P_build2(&b,np,4,style,NULL,0);
    rates_t *rates=make_rates(np,nr,0);if(nr==1)rates->rates[0]=1.0;
    double la=trial?100:MLMinBranchLength,lb=trial?0.003:0.17;
    profile_t *ref=PosteriorProfile(&a.profile,&b.profile,la,lb,NULL,rates,np,0);
    double *sa=PSameVector(la,rates),*da=PDiffVector(sa,rates),*sb=PSameVector(lb,rates),*db=PDiffVector(sb,rates);
    double table[256],gw[257],gv[257*4];unsigned char gc[257];
    memcpy(table,sa,nr*sizeof(double));memcpy(table+nr,da,nr*sizeof(double));memcpy(table+2*nr,sb,nr*sizeof(double));memcpy(table+3*nr,db,nr*sizeof(double));
    FtGpuProfileView av=FtProfileViewOf(&a.profile),bv=FtProfileViewOf(&b.profile);
    int status=ft_gpu_jc_posterior(g_ctx,np,nr,&av,&bv,rates->ratecat,table,gw,gc,gv);
    if(status!=FT_GPU_OK)case_fail("status %d",status);
    else {
      if(memcmp(gw,ref->weights,sizeof(gw))||memcmp(gc,ref->codes,sizeof(gc)))case_fail("codes/weights not bitwise equal");
      int v=0;for(int i=0;i<np;i++)if(ref->weights[i]>0&&ref->codes[i]==NOCODE){if(memcmp(gv+4*i,ref->vectors+4*v,4*sizeof(double)))case_fail("vector not bitwise equal at %d",i);v++;}
      if(v!=ref->nVectors)case_fail("vector count");
    }
    /* Invalid tables/categories must not write ANY output. */
    double old=table[0];table[0]=NAN;
    memset(gw,0x5a,sizeof(gw));memset(gv,0x5a,sizeof(gv));memset(gc,0x5a,sizeof(gc));
    if(ft_gpu_jc_posterior(g_ctx,np,nr,&av,&bv,rates->ratecat,table,gw,gc,gv)!=FT_GPU_ERR_INVALID)case_fail("NaN table accepted");table[0]=old;
    unsigned int oldcat=rates->ratecat[0];rates->ratecat[0]=nr;
    if(ft_gpu_jc_posterior(g_ctx,np,nr,&av,&bv,rates->ratecat,table,gw,gc,gv)!=FT_GPU_ERR_INVALID)case_fail("invalid rate accepted");rates->ratecat[0]=oldcat;
    for(size_t i=0;i<sizeof(gw);i++)if(((unsigned char*)gw)[i]!=0x5a){case_fail("weights changed on failure");break;}
    for(size_t i=0;i<sizeof(gv);i++)if(((unsigned char*)gv)[i]!=0x5a){case_fail("vectors changed on failure");break;}
    for(size_t i=0;i<sizeof(gc);i++)if(gc[i]!=0x5a){case_fail("codes changed on failure");break;}
    if(rc==0&&style==0&&trial==0){
      FtGpuConfig cfg={0};cfg.device_id=-1;cfg.memory_limit_bytes=32;FtGpuContext *small=NULL;
      if(ft_gpu_create(&cfg,&small)!=FT_GPU_OK)case_fail("budget context create");
      else {if(ft_gpu_jc_posterior(small,np,nr,&av,&bv,rates->ratecat,table,gw,gc,gv)!=FT_GPU_ERR_MEMORY)case_fail("budget not enforced");FtGpuStats st;ft_gpu_stats_get(small,&st);if(st.deviceBytes!=0)case_fail("failed allocation leaked");ft_gpu_destroy(small);}
    }
    /* Successful call after invalid input also exercises cache reuse. */
    if(ft_gpu_jc_posterior(g_ctx,np,nr,&av,&bv,rates->ratecat,table,gw,gc,gv)!=FT_GPU_OK)case_fail("recovery failed");
    ft_gpu_profile_invalidate(g_ctx,av.owner,av.serial);ft_gpu_profile_invalidate(g_ctx,bv.owner,bv.serial);
    FreeProfile(ref,np,0);P_free(&a);P_free(&b);free_rates(rates);
    myfree(sa,nr*sizeof(double));myfree(da,nr*sizeof(double));myfree(sb,nr*sizeof(double));myfree(db,nr*sizeof(double));
    if(g_fail==before)case_pass();
  }
}

int main(int argc, char **argv) {
  const char *alignment = NULL;
  int i, require_gpu = 0;
  for (i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--require-gpu") == 0) {
      require_gpu = 1;
    } else if (strcmp(argv[i], "--alignment") == 0 && i + 1 < argc) {
      alignment = argv[++i];
    } else if (strcmp(argv[i], "--list") == 0) {
      g_list = 1;
    } else if (strcmp(argv[i], "--verbose") == 0) {
      g_verbose = 1;
    } else if (strcmp(argv[i], "--seq-ref") == 0) {
      g_seqRef = 1;
    } else if (strcmp(argv[i], "--nt") == 0) {
      g_realNt = 1;
    } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
      g_rng = (uint64_t)atoll(argv[++i]) * 2862933555777941757ULL + 3037000493ULL;
      if (g_rng == 0) g_rng = 88172645463325252ULL;
    } else {
      fprintf(stderr, "test_kernels: unknown option %s\n", argv[i]);
      return 2;
    }
  }
  (void)g_verbose;

  showProgress = 0;
  if (ft_gpu_is_available()) {
    FtGpuConfig cfg;
    memset(&cfg, 0, sizeof(cfg));
    cfg.device_id = -1;
    cfg.stats_level = 1;
    cfg.debug_sequential = g_seqRef;
    if (ft_gpu_create(&cfg, &g_ctx) != FT_GPU_OK) g_ctx = NULL;
    if (g_ctx != NULL) {
      char info[256];
      ft_gpu_device_info(g_ctx, info, sizeof(info));
      printf("# device: %s\n", info);
    }
  }
  if (g_ctx == NULL)
    printf("# no CUDA device / CPU-only build: GPU cases will be skipped\n");

  if (require_gpu && !g_ctx) { fprintf(stderr, "CUDA device required\n"); return 1; }

  /* protein pass */
  nCodes = 20;
  codesString = codesStringAA;
  useMatrix = true;
  test_vector_index();
  test_pair_loglk_protein();
  test_dist_synthetic();

  /* nucleotide pass */
  nCodes = 4;
  codesString = codesStringNT;
  test_pair_loglk_dna();
  test_vector_index();

  /* real alignment (--nt selects the nucleotide alphabet) */
  if (g_realNt) {
    nCodes = 4;
    codesString = codesStringNT;
  } else {
    nCodes = 20;
    codesString = codesStringAA;
  }
  if (alignment != NULL) test_real_alignment(alignment);

  test_api_errors();
  test_resident_matrix(4);
  test_resident_matrix(20);
  test_jc_posterior();
  test_fallback_policy();

  printf("# cases passed=%d failed=%d skipped=%d\n", g_pass, g_fail, g_skip);
  printf("# checks: scalar=%ld site=%ld dist=%ld; worst ratio to tolerance: scalar=%.3g "
         "site=%.3g; worst relative deviation of a scalar: %.3g\n",
         g_nScalarChecks, g_nSiteChecks, g_nDistChecks, g_worstScalar, g_worstSite,
         g_worstScalarRel);
  if (g_ctx != NULL) {
    FtGpuStats s;
    ft_gpu_stats_get(g_ctx, &s);
    printf("# gpu: uploads=%llu hits=%llu evictions=%llu pair_jobs=%llu site_evals=%llu "
           "launches=%llu h2d=%llu d2h=%llu peak_device=%llu\n",
           (unsigned long long)s.nProfileUploads, (unsigned long long)s.nProfileHits,
           (unsigned long long)s.nProfileEvictions, (unsigned long long)s.nPairsGpu,
           (unsigned long long)s.nSitesGpu, (unsigned long long)s.nKernelLaunches,
           (unsigned long long)s.bytesH2D, (unsigned long long)s.bytesD2H,
           (unsigned long long)s.peakDeviceBytes);
    ft_gpu_destroy(g_ctx);
    g_ctx = NULL;
  }
  return g_fail == 0 ? 0 : 1;
}
