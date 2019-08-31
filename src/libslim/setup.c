/*!
\file
\brief Functions that deal with setting up the matrices

\date   Started 3/9/2015
\author George & Xia
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/*************************************************************************/
/*! @brief Sorts the indices in increasing order
    @param mat  the matrix itself
    @param what is either GK_CSR_ROW or GK_CSR_COL indicating which set of
           indices to sort.
*/
/**************************************************************************/
void slim_csr_SortIndices(gk_csr_t *mat, int what)
{
  int n, nn=0;
  ssize_t *ptr;
  int *ind;
  float *val;

  switch (what) {
    case GK_CSR_ROW:
      if (!mat->rowptr)
        gk_errexit(SIGERR, "Row-based view of the matrix does not exists.\n");

      n   = mat->nrows;
      ptr = mat->rowptr;
      ind = mat->rowind;
      val = mat->rowval;
      break;

    case GK_CSR_COL:
      if (!mat->colptr)
        gk_errexit(SIGERR, "Column-based view of the matrix does not exists.\n");

      n   = mat->ncols;
      ptr = mat->colptr;
      ind = mat->colind;
      val = mat->colval;
      break;

    default:
      gk_errexit(SIGERR, "Invalid index type of %d.\n", what);
      return;
  }

  #pragma omp parallel if (n > 100)
  {
    ssize_t i, j, k;
    gk_ikv_t *cand;
    float *tval;

    #pragma omp single
    for (i=0; i<n; i++) 
      nn = gk_max(nn, ptr[i+1]-ptr[i]);
  
    cand = gk_ikvmalloc(nn, "gk_csr_SortIndices: cand");
    if (val) {
      tval = gk_fmalloc(nn, "gk_csr_SortIndices: tval");
    }
  
    #pragma omp for schedule(static)
    for (i=0; i<n; i++) {
      for (k=0, j=ptr[i]; j<ptr[i+1]; j++) {
        if (j > ptr[i] && ind[j] < ind[j-1])
          k = 1; /* an inversion */
        cand[j-ptr[i]].val = j-ptr[i];
        cand[j-ptr[i]].key = ind[j];
        if (val) {
          tval[j-ptr[i]] = val[j];
        }
      }
      if (k) {
        gk_ikvsorti(ptr[i+1]-ptr[i], cand);
        for (j=ptr[i]; j<ptr[i+1]; j++) {
          ind[j] = cand[j-ptr[i]].key;
          if (val) {
            val[j] = tval[cand[j-ptr[i]].val];
          }
        }
      }
    }
    if (val) {
      gk_free((void **)&cand, &tval, LTERM);
    } else {
      gk_free((void **)&cand, LTERM);
    }
  }
}

/*************************************************************************/
/*! @brief This function sets up the training matrix from the user's input 
    
    @param  params hyper-parameters for training
            nrows  number of rows of the matrix
            rowptr row pointer of the matrix
            rowind row indices of the matrix
            rowval row value of the matrix

    @return mat a gk_csr matrix which contains both row and column
                representations
*/
/*************************************************************************/
gk_csr_t *CreateTrainingMatrix(params_t *params, int32_t nrows, ssize_t *rowptr,
                               int32_t *rowind, float *rowval) {
  gk_csr_t *mat;

  /* allocate the matrix and fill in the fields */
  mat = gk_csr_Create();

  mat->nrows = nrows;
  mat->ncols = gk_i32max(rowptr[nrows], rowind, 1) + 1;

  mat->rowptr = gk_zcopy(nrows + 1, rowptr, gk_zmalloc(nrows + 1, "rowptr"));
  mat->rowind =
      gk_i32copy(rowptr[nrows], rowind, gk_i32malloc(rowptr[nrows], "rowind"));
  if (rowval)
    mat->rowval =
        gk_fcopy(rowptr[nrows], rowval, gk_fmalloc(rowptr[nrows], "rowval"));
  else
    mat->rowval = NULL;

  gk_csr_CreateIndex(mat, GK_CSR_COL);

  gk_csr_ComputeNorms(mat, GK_CSR_COL);

  slim_csr_SortIndices(mat, GK_CSR_COL);

  return mat;
}
