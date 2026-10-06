/* SPDX-License-Identifier: GPL-3.0-or-later */
/* profile_pack.h -- profile representation helpers shared by the CPU and CUDA
 * builds.
 *
 * The compressed FastTree profile stores a state vector only for positions
 * with weights[site] > 0 && codes[site] == NOCODE, and GET_FREQ walks them with
 * an implicit cursor.  The backend needs that cursor as an explicit index so
 * that independent threads can find a site's vector, so the mapping is built
 * once per uploaded profile and verified against the original rule.
 */
#ifndef FT_PROFILE_PACK_H
#define FT_PROFILE_PACK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Builds index[nPos] where index[i] is the vector ordinal of site i, or -1 if
   the site has no stored vector.  *nVectorsOut receives the number of vectors.
   Returns 0 on success, non-zero on error (NULL arguments or negative nPos). */
int ft_pack_vector_index(const double *weights, const unsigned char *codes, int nPos,
                         int *index, int *nVectorsOut);

/* Number of vectors implied by the GET_FREQ rule. */
int ft_count_vectors(const double *weights, const unsigned char *codes, int nPos);

/* The GET_FREQ predicate, exposed for tests and callers that must stay
   bit-compatible with FastTree.c. */
#define FT_HAS_VECTOR(weights, codes, i) \
  ((weights)[i] > 0 && (codes)[i] == 127 /* NOCODE */)

#ifdef __cplusplus
}
#endif

#endif /* FT_PROFILE_PACK_H */
