/*!
\file
\brief Various prediction functions

\date   Started 3/10/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/**************************************************************************/
/*! Get the top-N recommendations given the provided historical data */
/**************************************************************************/
int32_t GetRecommendations(params_t *params, gk_csr_t *smat, int32_t nratings,
                           int32_t *itemids, float *ratings, int32_t nrcmds,
                           int32_t *rids, float *rscores) {
  ssize_t j;
  int32_t iR, i, k, ncols, ncand;
  ssize_t *rowptr;
  int32_t *rowind, *marker;
  float *rowval, rating;
  gk_fkv_t *cand;

  ncols = smat->ncols;
  rowptr = smat->rowptr;
  rowind = smat->rowind;
  rowval = smat->rowval;

  marker = gk_i32smalloc(ncols, -1, "marker");
  cand = gk_fkvmalloc(ncols, "cand");

  /* mark the already rated so that they will not be used */
  for (iR = 0; iR < nratings; iR++) {
    if (itemids[iR] < ncols && itemids[iR] >= 0)
      marker[itemids[iR]] = -2;
  }

  ncand = 0;
  for (iR = 0; iR < nratings; iR++) {
    i = itemids[iR];
    if (i >= ncols && i < 0)
      continue;

    rating = (ratings ? ratings[iR] : 1.0);
    for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
      k = rowind[j];
      if (marker[k] == -2)
        continue; /* part of the history */

      if (marker[k] == -1) {
        cand[ncand].val = k;
        cand[ncand].key = 0.0;
        marker[k] = ncand++;
      }
      cand[marker[k]].key += rating * rowval[j];
    }
  }

  gk_fkvsortd(ncand, cand);

  nrcmds = gk_min(ncand, nrcmds);
  for (iR = 0; iR < nrcmds; iR++) {
    rids[iR] = cand[iR].val;
    rscores[iR] = cand[iR].key;
  }

  gk_free((void **)&marker, &cand, LTERM);

  return nrcmds;
}
