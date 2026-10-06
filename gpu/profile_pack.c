/* SPDX-License-Identifier: GPL-3.0-or-later */
/* profile_pack.c -- see profile_pack.h */
#include "profile_pack.h"

int ft_count_vectors(const double *weights, const unsigned char *codes, int nPos) {
  int i;
  int n = 0;
  if (weights == 0 || codes == 0 || nPos < 0) return -1;
  for (i = 0; i < nPos; i++) {
    if (FT_HAS_VECTOR(weights, codes, i)) n++;
  }
  return n;
}

int ft_pack_vector_index(const double *weights, const unsigned char *codes, int nPos,
                         int *index, int *nVectorsOut) {
  int i;
  int n = 0;
  if (weights == 0 || codes == 0 || index == 0 || nVectorsOut == 0 || nPos < 0)
    return 1;
  for (i = 0; i < nPos; i++) {
    if (FT_HAS_VECTOR(weights, codes, i))
      index[i] = n++;
    else
      index[i] = -1;
  }
  *nVectorsOut = n;
  return 0;
}
