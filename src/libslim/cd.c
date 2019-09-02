/*!
\file
\brief The routines associated with model estimation

\date   Started 3/9/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/**************************************************************************/
/*! @brief  implementation of axpy.

    @param  wspace workspace that contains the sparse user-item matrix
            iI     item id that specify the column x of the matrix to be
                   added to yhat
            xi     coefficient of x
            yhat   a dense vector

    @return void
*/
/**************************************************************************/
void AddSpVec(wspace_t *wspace, int32_t iI, double xi, double *yhat) {
  int32_t i;

  if (xi > EPSILON || xi < -EPSILON) {
    if (wspace->colval) {
      for (i = wspace->colptr[iI]; i < wspace->colptr[iI + 1]; i++) {
        yhat[wspace->colind[i]] += xi * wspace->colval[i];
      }
    } else {
      for (i = wspace->colptr[iI]; i < wspace->colptr[iI + 1]; i++) {
        yhat[wspace->colind[i]] += xi;
      }
    }
  }
}

/**************************************************************************/
/*! @brief  dot product of one column of the user-item matrix
            with a dense vector

    @param  wspace workspace that contains the sparse user-item matrix
            iI     item id that specify the column x of the matrix
            yhat   a dense vector

    @return res    the inner product
*/
/**************************************************************************/
double SpVecInnerProduct(wspace_t *wspace, int32_t iI, double *yhat) {
  int32_t i;
  double res = 0.0;

  if (wspace->colval) {
    for (i = wspace->colptr[iI]; i < wspace->colptr[iI + 1]; i++) {
      res += wspace->colval[i] * yhat[wspace->colind[i]];
    }
  } else {
    for (i = wspace->colptr[iI]; i < wspace->colptr[iI + 1]; i++) {
      res += yhat[wspace->colind[i]];
    }
  }
  return res;
}

/**************************************************************************/
/*! @brief  shuffle an fkv list of size n

    @param  list fkv list to be shuffled
            n    size of the fkv list

    @return void
*/
/**************************************************************************/
void ShuffleList(gk_fkv_t *list, int32_t n) {
  int32_t i;

  for (i = 0; i < n; i++) {
    gk_fkv_t temp = list[i];
    int32_t index = rand() % n;

    list[i] = list[index];
    list[index] = temp;
  }
}

/**************************************************************************/
/*! @brief  Coordinate Descent Algorithm for SLIM

    @param  wspace workspace that contains params and active columns
            x      weight vector to be estimated
            y      target to be approximated
            yhat   approximation of y
            nacols number of active items to be trained on

    @return rstatus convergence status, if the algorithm converges within
            maxniters, rstatus=1, else 0
*/
/**************************************************************************/
int32_t CoordinateDescent(wspace_t *wspace, double *x, double *y, double *yhat,
                          int32_t nacols) {
  int32_t i, t, iI, nnz, rstatus = 0;
  double dltx, aTy, aTa, xi, newxi, ip;
  double numerator;

  /* initialize yhat */
  for (i = 0; i < nacols; i++) {
    AddSpVec(wspace, wspace->cdacols[i].val, x[wspace->cdacols[i].val], yhat);
  }

  for (t = 0; t < wspace->maxniters; t++) {
    dltx = 0.0;

    ShuffleList(wspace->cdacols, nacols);
    for (i = 0; i < nacols; i++) {
      iI = wspace->cdacols[i].val;
      aTy = wspace->cdacols[i].key;
      aTa = wspace->cnorms[iI];
      xi = x[iI];

      AddSpVec(wspace, iI, -xi, yhat);
      ip = SpVecInnerProduct(wspace, iI, yhat);
      numerator = aTy - ip;
      newxi = numerator > wspace->params->l1r
                  ? (numerator - wspace->params->l1r) /
                        ((aTa * aTa) + wspace->params->l2r)
                  : 0.0;
      AddSpVec(wspace, iI, newxi, yhat);

      x[iI] = newxi;
      dltx += (newxi - xi) * (newxi - xi);
    }

    if (dltx < wspace->params->optTol) {
      rstatus = 1;
      break;
    }
  }
  wspace->niters = t + 1;
  return rstatus;
}
