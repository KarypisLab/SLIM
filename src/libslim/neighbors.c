/*!
\file
\brief Functions that deal with finding neighbors of columns for FSLIM
       type of models.

\date   Started 3/16/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/*************************************************************************/
/*! Finds the best neighbors of column iC. */
/*************************************************************************/
int32_t FindColumnNeighbors(params_t *params, wspace_t *wspace, gk_csr_t *mat,
                            int32_t iC) {
  ssize_t i, ii, j, k;
  int32_t ncand;
  ssize_t *rowptr, *colptr;
  int *rowind, *colind, *marker;
  float *rowval, *colval, *cnorms, cval;
  gk_fkv_t *cand;

  GKASSERT((rowptr = mat->rowptr) != NULL);
  GKASSERT((rowind = mat->rowind) != NULL);
  GKASSERT((colptr = mat->colptr) != NULL);
  GKASSERT((colind = mat->colind) != NULL);
  rowval = mat->rowval;
  colval = mat->colval;
  GKASSERT((cnorms = mat->cnorms) != NULL);

  if (colptr[iC] == colptr[iC + 1])
    return 0;

  GKASSERT((marker = wspace->marker) != NULL);
  GKASSERT((cand = wspace->cand) != NULL);

  switch (params->simtype) {
  case SLIM_SIMTYPE_DOTP:
    for (ncand = 0, ii = colptr[iC]; ii < colptr[iC + 1]; ii++) {
      i = colind[ii];
      cval = (colval ? colval[ii] : 1.0);

      for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
        if ((k = rowind[j]) == iC)
          continue;
        if (marker[k] == -1) {
          cand[ncand].val = k;
          cand[ncand].key = 0;
          marker[k] = ncand++;
        }
        if (rowval)
          cand[marker[k]].key += rowval[j] * cval;
        else
          cand[marker[k]].key += cval;
      }
    }

    break;

  case SLIM_SIMTYPE_COS:
    for (ncand = 0, ii = colptr[iC]; ii < colptr[iC + 1]; ii++) {
      i = colind[ii];
      cval = (colval ? colval[ii] : 1.0);

      for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
        if ((k = rowind[j]) == iC)
          continue;
        if (marker[k] == -1) {
          cand[ncand].val = k;
          cand[ncand].key = 0;
          marker[k] = ncand++;
        }
        if (rowval)
          cand[marker[k]].key += rowval[j] * cval;
        else
          cand[marker[k]].key += cval;
      }
    }

    for (i = 0; i < ncand; i++)
      cand[i].key = cand[i].key / cnorms[cand[i].val];

    break;

  case SLIM_SIMTYPE_JAC:
    for (ncand = 0, ii = colptr[iC]; ii < colptr[iC + 1]; ii++) {
      i = colind[ii];
      cval = (colval ? colval[ii] : 1.0);

      for (j = rowptr[i]; j < rowptr[i + 1]; j++) {
        if ((k = rowind[j]) == iC)
          continue;
        if (marker[k] == -1) {
          cand[ncand].val = k;
          cand[ncand].key = 0;
          marker[k] = ncand++;
        }
        if (rowval)
          cand[marker[k]].key += rowval[j] * cval;
        else
          cand[marker[k]].key += cval;
      }
    }

    for (i = 0; i < ncand; i++)
      cand[i].key =
          cand[i].key / (cnorms[cand[i].val] + cnorms[iC] - cand[i].key);
    break;

  default:
    errexit("Unknown similarity measure %d\n", params->simtype);
    return -1;
  }

  /* reset the marker array */
  for (i = 0; i < ncand; i++)
    marker[cand[i].val] = -1;

  gk_dfkvkselect(ncand, gk_min(params->nnbrs, ncand), cand);
  gk_fkvsortd(gk_min(params->nnbrs, ncand), cand);

  return gk_min(params->nnbrs, ncand);
}
