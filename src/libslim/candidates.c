/*!
\file
\brief Functions that deal with finding neighbors of columns for FSLIM
       type of models.

\date   Started 6/15/2019
\author
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/*************************************************************************
 *  Since it holds:
 *      [R'R]_ij = 0 => W_ij=0.
 *  Thus, even in the case we want to compute the complete SLIM only the
 *  relevant subset of the columns needs to be considered for optimization
 ***************************************************************************/
int32_t FindColumnCandidates(params_t *params, wspace_t *wspace, gk_csr_t *mat,
                             int32_t iC) {
  ssize_t i, ii, j, k;
  int32_t ncand;
  ssize_t *rowptr, *colptr;
  int *rowind, *colind, *marker;
  float *rowval, *colval, cval;
  gk_fkv_t *cand;

  GKASSERT((rowptr = mat->rowptr) != NULL);
  GKASSERT((rowind = mat->rowind) != NULL);
  GKASSERT((colptr = mat->colptr) != NULL);
  GKASSERT((colind = mat->colind) != NULL);
  rowval = mat->rowval;
  colval = mat->colval;

  if (colptr[iC] == colptr[iC + 1])
    return 0;

  GKASSERT((marker = wspace->marker) != NULL);
  GKASSERT((cand = wspace->cand) != NULL);

  /* Only the positive dot products will be considered */
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

  /* reset the marker array */
  for (i = 0; i < ncand; i++)
    marker[cand[i].val] = -1;

  return ncand;
}
