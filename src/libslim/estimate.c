/*!
\file
\brief The routines associated with model estimation
\date   Started 3/9/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

#ifdef USE_MKL
#include <mkl.h>

/**************************************************************************/
/*! @brief  Estimate a SLIM model using ADMM
    @param  params parameters and settings for training
            tmat   sparse user-item matrix
            imat   matrix to initialize the model, can be NULL
    @return mat    the SLIM model matrix
*/
/**************************************************************************/
#define SPARSE_CHECK_STATUS(function, error_message)                           \
  do {                                                                         \
    if (function != SPARSE_STATUS_SUCCESS) {                                   \
      printf(error_message);                                                   \
      fflush(0);                                                               \
      status = 1;                                                              \
      goto memory_free_admm;                                                   \
    }                                                                          \
  } while (0)

#define max(a, b)                                                              \
  ({                                                                           \
    __typeof__(a) _a = (a);                                                    \
    __typeof__(b) _b = (b);                                                    \
    _a > _b ? _a : _b;                                                         \
  })
gk_csr_t *EstimateModelADMM(params_t *params, gk_csr_t *tmat, gk_csr_t *imat) {
  /* Welcome message */
  printf("Learning the model using ADMM... \n");

  mkl_set_threading_layer(MKL_THREADING_GNU); // Needed when GCC is used

  /* local copies of problem dimensions */
  int32_t nnz = (int)tmat->rowptr[tmat->nrows];
  int m = (int)tmat->ncols;
  int n = (int)tmat->nrows;
  double RHO = 10000.0;
  int MAXITERS = 30;

  /* Declaration of Variables */

  // Sparse Matrices
  double *values_Rt = NULL, *values_R = NULL, *values_C = NULL;
  int *columns_Rt = NULL, *columns_R = NULL, *columns_C = NULL;
  int *rowIndex_Rt = NULL, *rowIndex_R = NULL;
  sparse_matrix_t csrRt = NULL, csrR = NULL;

  // Dense Matrices
  double *T = NULL, *A = NULL, *B = NULL, *C = NULL, *P = NULL, *W = NULL;

  // timers
  double s_initial, s_elapsed;

  // iterators, flags and NLA temps
  int i, j, status;
  int ipiv[m];

  /* Memory Allocation */

  // Sparse Matrices
  values_Rt = (double *)mkl_malloc(sizeof(double) * nnz, 128);
  columns_Rt = (int *)mkl_malloc(sizeof(int) * nnz, 128);
  rowIndex_Rt = (int *)mkl_malloc(sizeof(int) * (m + 1), 128);
  values_R = (double *)mkl_malloc(sizeof(double) * nnz, 128);
  columns_R = (int *)mkl_malloc(sizeof(int) * nnz, 128);
  rowIndex_R = (int *)mkl_malloc(sizeof(int) * (n + 1), 128);

  // Dense Matrices
  T = (double *)mkl_malloc(m * m * sizeof(double), 64);
  A = (double *)mkl_malloc(m * m * sizeof(double), 64);
  B = (double *)mkl_malloc(m * m * sizeof(double), 64);
  W = (double *)mkl_malloc(m * m * sizeof(double), 64);
  C = (double *)mkl_malloc(m * m * sizeof(double), 64);
  P = (double *)mkl_malloc(m * m * sizeof(double), 64);

  /* Variable Initialization */

  // Initialization of Sparse Matrices
  for (i = 0; i < nnz; i++)
    values_Rt[i] = (double)tmat->colval[i];
  for (i = 0; i < nnz; i++)
    columns_Rt[i] = (int)tmat->colind[i];
  for (i = 0; i < m + 1; i++)
    rowIndex_Rt[i] = (int)tmat->colptr[i];
  for (i = 0; i < nnz; i++)
    values_R[i] = (double)tmat->rowval[i];
  for (i = 0; i < nnz; i++)
    columns_R[i] = (int)tmat->rowind[i];
  for (i = 0; i < n + 1; i++)
    rowIndex_R[i] = (int)tmat->rowptr[i];

  // Create handles for matrices Rt and R stored in CSR format
  SPARSE_CHECK_STATUS(mkl_sparse_d_create_csr(&csrRt, SPARSE_INDEX_BASE_ZERO, m,
                                              n, rowIndex_Rt, rowIndex_Rt + 1,
                                              columns_Rt, values_Rt),
                      "Error after MKL_SPARSE_D_CREATE_CSR, csrRt \n");
  SPARSE_CHECK_STATUS(mkl_sparse_d_create_csr(&csrR, SPARSE_INDEX_BASE_ZERO, n,
                                              m, rowIndex_R, rowIndex_R + 1,
                                              columns_R, values_R),
                      "Error after MKL_SPARSE_D_CREATE_CSR, csrR \n");

  // Initialization of Dense Matrices
  for (i = 0; i < m * m; i++)
    T[i] = W[i] = A[i] = B[i] = C[i] = P[i] = 0.0;

  /* Computation Starts */

  // T = Rt * R
  if (params->dbglvl == SLIM_DBG_TIME) { // Detailed time reporting
    printf("Building matrix RtR...."), fflush(0);
    s_initial = dsecnd();
  }
  mkl_sparse_d_spmmd(SPARSE_OPERATION_NON_TRANSPOSE, csrRt, csrR,
                     SPARSE_LAYOUT_ROW_MAJOR, T, m);
  if (params->dbglvl == SLIM_DBG_TIME) { // Detailed time reporting
    s_elapsed = dsecnd() - s_initial;
    printf(".... completed in %.5f seconds\n", s_elapsed);
  }

  if (params->dbglvl == SLIM_DBG_TIME) { // Detailed reporting
    long int sum = 0;
    for (i = 0; i < m * m; i++)
      if (T[i] > params->l1r)
        sum++;
    printf("Density of RtR=%lf\n", sum / (double)(m * m));
  }

  // copy RtR to P
  mkl_domatcopy('R', 'N', m, m, 1.0, T, m, P, m);

  // add l2r+rho in the diagonal
  for (i = 0; i < m; i++)
    P[i * m + i] += params->l2r + RHO;

  if (params->dbglvl == SLIM_DBG_TIME) { // Detailed time reporting
    printf("Factorizing and Inverting RtR+(l2+rho)I...."), fflush(0);
    s_initial = dsecnd();
  }
  // Factorization of P using the Bunch-Kaufmann Decomposition
  // LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'U', m, P, m, ipiv);
  LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', m, P, m);
  // Inversion of P
  // LAPACKE_dsytri(LAPACK_ROW_MAJOR, 'U', m, P, m, ipiv);
  LAPACKE_dpotri(LAPACK_ROW_MAJOR, 'U', m, P, m);
  if (params->dbglvl == SLIM_DBG_TIME) { // Detailed time reporting
    s_elapsed = dsecnd() - s_initial;
    printf(".... completed in %.5f seconds\n", s_elapsed);
  }
  // Form the full inverse
  for (i = 0; i < m; i++) {
    for (j = 0; j < i; j++)
      P[i * m + j] = P[j * m + i];
  }

  // openblas...
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, P, m, T,
              m, 0.0, A, m);

  // the core ADMM iterations
  for (int iter = 0; iter < MAXITERS; iter++) {
    // W := rho*W
    cblas_dscal(m * m, RHO, W, 1);

    // W :=  W-C
    cblas_daxpy(m * m, -1.0, C, 1, W, 1);

    // T := P*W
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m, 1.0, P, m,
                W, m, 0.0, T, m);

    // T = T+A
    cblas_daxpy(m * m, 1.0, A, 1, T, 1);

    double gamma[m];
    for (i = 0; i < m; i++)
      gamma[i] = T[i * m + i] / P[i * m + i];

    // B = -P*diagMat(gamma)
    for (i = 0; i < m; i++) {
      for (j = 0; j < m; j++)
        B[i * m + j] = -1.0 * P[i * m + j] * gamma[j];
    }

    // B := B + T
    cblas_daxpy(m * m, 1.0, T, 1, B, 1);

    // soft thresholding
    double irho = 1.0 / RHO, kappa = (params->l1r) / RHO;
    for (int i = 0; i < m * m; i++) {
      double alpha = B[i] + irho * C[i];
      double temp = max(alpha - kappa, 0.0) - max(-alpha - kappa, 0.0);
      W[i] = max(temp, 0.0);
    }

    // B := B - W
    cblas_daxpy(m * m, -1.0, W, 1, B, 1);

    // B := rho*B
    cblas_dscal(m * m, RHO, B, 1);

    // C := C + B
    cblas_daxpy(m * m, 1.0, B, 1, C, 1);
  }

  size_t modelnnz = 0;
  for (i = 0; i < m * m; i++) {
    if (W[i] > 0.0)
      modelnnz++;
  }

  if (params->dbglvl == SLIM_DBG_TIME) { // Detailed reporting
    printf("Density of the model = %3.3f\n", modelnnz / (double)(m * m));
  }

  /* Return in gk_csr_t */
  gk_csr_t *mat = NULL;
  size_t nrows = (size_t)m, ncols = (size_t)m;
  ssize_t *rowptr;
  int *rowind, *iinds, *jinds, ipos = 0;
  float *rowval = NULL, *vals;

  /* read the data into three arrays */
  iinds = gk_i32malloc(modelnnz, "iinds");
  jinds = gk_i32malloc(modelnnz, "jinds");
  vals = gk_fmalloc(modelnnz, "vals");

  /* read the data into three arrays */
  for (i = 0; i < m; i++) {
    for (j = 0; j < m; j++) {
      if (W[i * m + j] > 0.0) {
        iinds[ipos] = i;
        jinds[ipos] = j;
        vals[ipos] = W[i * m + j];
        ipos++;
      }
    }
  }

  /* convert (i, j, v) into a CSR matrix */
  mat = gk_csr_Create();
  mat->nrows = nrows;
  mat->ncols = ncols;
  rowptr = mat->rowptr = gk_zsmalloc(nrows + 1, 0, "rowptr");
  rowind = mat->rowind = gk_i32malloc(modelnnz, "rowind");
  rowval = mat->rowval = gk_fmalloc(modelnnz, "rowval");

  for (i = 0; i < modelnnz; i++) {
    rowptr[iinds[i]]++;
  }

  MAKECSR(i, nrows, rowptr);

  for (i = 0; i < modelnnz; i++) {
    rowind[rowptr[iinds[i]]] = jinds[i];
    rowval[rowptr[iinds[i]]] = vals[i];
    rowptr[iinds[i]]++;
  }
  SHIFTCSR(i, nrows, rowptr);

  /* Deallocate memory */
memory_free_admm:
  // Release sparse matrix handles and deallocate arrays
  if (mkl_sparse_destroy(csrRt) != SPARSE_STATUS_SUCCESS) {
    printf(" Error after MKL_SPARSE_DESTROY, csrRt \n");
    fflush(0);
    status = 1;
  }
  mkl_free(values_Rt);
  mkl_free(columns_Rt);
  mkl_free(rowIndex_Rt);

  if (mkl_sparse_destroy(csrR) != SPARSE_STATUS_SUCCESS) {
    printf(" Error after MKL_SPARSE_DESTROY, csrR \n");
    fflush(0);
    status = 1;
  }
  mkl_free(values_R);
  mkl_free(columns_R);
  mkl_free(rowIndex_R);

  // Deallocate Dense Matrices
  mkl_free(T);
  mkl_free(A);
  mkl_free(B);
  mkl_free(P);
  mkl_free(C);
  mkl_free(W);

  // Deallocate temporary arrays used in preparing the returning gk_csr_t struct
  gk_free((void **)&iinds, &jinds, &vals, LTERM);

  return mat;
}

#endif

#ifndef USE_MKL
gk_csr_t *EstimateModelADMM(params_t *params, gk_csr_t *tmat, gk_csr_t *imat) {
  /* Welcome message */
  printf("You cannot use ADMM without MKL! \n\n\n");
  printf("Use the \"-algo=cd\" flag instead! \n\n\n");
  printf("Exiting!\n");
  printf(
      "------------------------------------------------------------------\n");
  exit(0);
}
#endif

/**************************************************************************/
/*! @brief  Estimate a SLIM/FSLIM model using coordinate descent (CD)
    @param  params parameters and settings for training
            tmat   sparse user-item matrix
            imat   matrix to initialize the model, can be NULL
    @return mat    the SLIM model matrix
*/
/**************************************************************************/
gk_csr_t *EstimateModelCD(params_t *params, gk_csr_t *tmat, gk_csr_t *imat) {
  printf("Using Coordinate Descent! \n");
  /* number of nonzero elements in the final model W */
  ssize_t tnnz;

  /* reference of the rating matrix */
  int32_t iC, iI, nrows, ncols, idx;
  ssize_t *colptr;
  int32_t *colind;
  float *colval;
  float *cnorms;

  /* final model W */
  gk_csr_t *mat = NULL;

  /* training variables */
  double l1r, l2r, error, objval;

  /* intermediate storage for cols in each thread */
  int32_t *nnzs;
  gk_fkv_t **lists;

  /* initialize the reference */
  nrows = tmat->nrows;
  ncols = tmat->ncols;
  colptr = tmat->colptr;
  colind = tmat->colind;
  colval = tmat->colval;
  cnorms = tmat->cnorms;

  if (!cnorms) {
    printf("Norms of cols not calculated! \n");
  }

  /* allocate memory for the head-points of the results */
  nnzs = gk_i32smalloc(ncols, 0, "nnzs");
  lists = (gk_fkv_t **)gk_malloc(ncols * sizeof(gk_fkv_t *), "lists");
  memset(lists, 0, ncols * sizeof(gk_fkv_t *));

  error = 0.0;  /* the total 1/2 squared error of the approximation */
  objval = 0.0; /* the total value of the objective */

/* start estimating each column of the model in parallel */
#pragma omp parallel default(shared), reduction(+                 \
                                                : error, objval), \
    num_threads(params->nthreads)
  {
    ssize_t i, j;
    int32_t nnz, nnbrs, rstatus;
    double *y, *x, *yhat, *ATy, soln_rNorm, soln_obj, tmr = 0.0;
    wspace_t *wspace;
    gk_fkv_t *list;

    /* allocate per-thread memory */
    x = gk_dsmalloc(ncols, 0.0, "x");
    y = gk_dsmalloc(nrows, 0.0, "y");
    yhat = gk_dsmalloc(nrows, 0.0, "yhat");
    ATy = gk_dsmalloc(ncols, 0.0, "ATy");

    wspace = (wspace_t *)gk_malloc(sizeof(wspace_t), "wspace");
    memset(wspace, 0, sizeof(wspace_t));

    wspace->params = params;
    wspace->colptr = colptr;
    wspace->colind = colind;
    wspace->colval = colval;
    wspace->cnorms = cnorms;

    if (params->mtype == SLIM_MTYPE_FSLIM) {
      wspace->marker = gk_i32smalloc(ncols, -1, "marker");
      wspace->cand = gk_fkvmalloc(ncols, "cand");
    }

/* split the work among the threads */
#pragma omp for schedule(dynamic, 32)
    for (iC = 0; iC < ncols; iC++) {

      /* set the target vector */
      for (i = colptr[iC]; i < colptr[iC + 1]; i++) {
        y[colind[i]] = (colval ? colval[i] : 1.0);
      }

      /* calculate the mat-vec multiplication ATy */
      nnbrs = 0;
      for (i = 0; i < ncols; i++) {
        double ip = 0.0;
        for (j = colptr[i]; j < colptr[i + 1]; j++) {
          ip += colval ? colval[j] * y[colind[j]] : y[colind[j]];
        }
        ATy[i] = ip;
        if (ip > params->l1r && i != iC) {
          nnbrs++;
        }
      }

      /* deal with FSLIM */
      if (params->mtype == SLIM_MTYPE_FSLIM) {
        nnbrs = FindColumnNeighbors(params, wspace, tmat, iC);
        wspace->cdacols = gk_fkvmalloc(nnbrs, "cdacols");

        for (i = 0; i < nnbrs; i++) {
          wspace->cdacols[i].val = wspace->cand[i].val;
          wspace->cdacols[i].key = ATy[wspace->cand[i].val];
        }
      } else {
        wspace->cdacols = gk_fkvmalloc(nnbrs, "cdacols");
        for (i = 0, j = 0; i < ncols; i++) {
          if (ATy[i] > params->l1r && i != iC) {
            wspace->cdacols[j].val = i;
            wspace->cdacols[j].key = ATy[i];
            j++;

            /* flag active columns for solution initialization */
            x[i] = -0.1;
          }
        }
      }

      /* set the max # of cd iterations in an adaptive setting based
         on the number of target non-zeros */
      wspace->maxniters =
          gk_min(50 * (colptr[iC + 1] - colptr[iC]), params->maxniters);
      wspace->niters = 0;

      /* initialize the solution */
      if (imat != NULL) {
        /* use imat's corresponding column to initialize the solution */
        for (i = imat->colptr[iC]; i < imat->colptr[iC + 1]; i++) {
          /* only initialize active cols (x[i] == -0.1 in line 425) */
          x[imat->colind[i]] = x[imat->colind[i]] < 0. ? imat->colval[i] : 0.0;
        }

        /* set all flags to 0.0 if not initialized */
        for (i = 0; i < nnbrs; i++) {
          idx = wspace->cdacols[i].val;
          x[idx] = x[idx] < 0. ? 0.0 : x[idx];
        }
      } else {
        /* initialze weights of the active columns by 0.0 */
        for (i = 0; i < nnbrs; i++) {
          idx = wspace->cdacols[i].val;
          x[idx] = 0.0;
        }
      }

      /* solve the optimization problem using coordinate descent */
      rstatus = CoordinateDescent(wspace, x, y, yhat, nnbrs);

      /* calculate the err and obj */
      soln_rNorm = 0.0;
      for (i = 0; i < nrows; i++) {
        soln_rNorm += (y[i] - yhat[i]) * (y[i] - yhat[i]);
      }
      soln_rNorm = 0.5 * soln_rNorm; // 1/2 * ||r||_2^2
      error += soln_rNorm;

      soln_obj = soln_rNorm;
      for (i = 0; i < ncols; i++) {
        soln_obj +=
            (0.5 * params->l2r * (x[i] * x[i]) + params->l1r * fabs(x[i]));
      }
      objval += soln_obj; // 1/2 * ||r||_2^2 + l2r/2 * ||x||_2^2 + l1r * ||x||_1

      /* save the solution */
      for (nnz = 0, i = 0; i < ncols; i++) {
        if (fabs(x[i]) > EPSILON)
          nnz++;
      }
      list = gk_fkvmalloc(nnz, "list");
      for (nnz = 0, i = 0; i < ncols; i++) {
        if (fabs(x[i]) > EPSILON) {
          list[nnz].key = x[i];
          list[nnz].val = i;
          nnz++;
        }
      }
      nnzs[iC] = nnz;
      lists[iC] = list;

      IFSET(params->dbglvl, SLIM_DBG_PROGRESS,
            printf("Col: %5" PRId32 " %5zd rs: %3" PRId32 " nits: %4" PRId32
                   " nnz: %4" PRId32 " rsd: %.2le obj: %.2le ff: %.3lf nrm1: "
                   "%.3lf a0s: %.3lf tmr: %.2le\n",
                   iC, colptr[iC + 1] - colptr[iC], rstatus, wspace->niters,
                   nnz, soln_rNorm, soln_obj, soln_rNorm / soln_obj,
                   gk_dsum(ncols, x, 1), ComputeAvgZeroScore(tmat, x, y, 10),
                   gk_getwctimer(tmr)));

      /* restore the defaults for the next iteration */
      gk_free((void **)&(wspace->cdacols), LTERM);
      wspace->cdacols = NULL;

      for (i = colptr[iC]; i < colptr[iC + 1]; i++) {
        y[colind[i]] = 0.0;
      }

      for (i = 0; i < ncols; i++) {
        x[i] = 0.0;
      }

      for (i = 0; i < nrows; i++) {
        yhat[i] = 0.0;
      }
    }

    /* cleanup */
    if (params->mtype == SLIM_MTYPE_FSLIM) {
      gk_free((void **)&(wspace->cdacols), &(wspace->marker), &(wspace->cand),
              &wspace, &x, &y, &yhat, &ATy, LTERM);
    } else {
      gk_free((void **)&(wspace->cdacols), &wspace, &x, &y, &yhat, &ATy, LTERM);
    }
  }

  /* create and store the combined model matrix */
  tnnz = gk_i32sum(ncols, nnzs, 1);
  mat = SaveModel(tnnz, ncols, ncols, nnzs, lists);

  /* free the memory */
  for (iC = 0; iC < ncols; iC++) {
    gk_free((void **)&lists[iC], LTERM);
  }
  gk_free((void **)&nnzs, &lists, LTERM);

  IFSET(params->dbglvl, SLIM_DBG_INFO,
        printf("Done estimation: loss: %.5le, fit: %.5le, ffrac: %.3lf,  #nzs: "
               "%zd\n",
               objval, error, error / objval, tnnz));

  return mat;
}

/**************************************************************************/
/*! @brief  store the learned matrix.
    @param  tnnz   number of non-zeros elements of the matrix
            nrows  number of rows of the matrix
            ncols  number of columns of the matrix
            nnzs   number of non-zeros elements per column
            lists  learned matrix (columnwise)
    @return mat    matrix to store the model
*/
/**************************************************************************/
gk_csr_t *SaveModel(ssize_t tnnz, int32_t nrows, int32_t ncols, int32_t *nnzs,
                    gk_fkv_t **lists) {
  gk_csr_t *mat = NULL;
  int32_t iC, iI;

  mat = gk_csr_Create();
  mat->nrows = nrows;
  mat->ncols = ncols;
  mat->colptr = gk_zmalloc(ncols + 1, "colptr");
  mat->colind = gk_i32malloc(tnnz, "colind");
  mat->colval = gk_fmalloc(tnnz, "colval");

  mat->colptr[0] = 0;
  for (tnnz = 0, iC = 0; iC < ncols; iC++) {
    for (iI = 0; iI < nnzs[iC]; iI++, tnnz++) {
      mat->colind[tnnz] = lists[iC][iI].val;
      mat->colval[tnnz] = lists[iC][iI].key;
    }
    mat->colptr[iC + 1] = tnnz;
  }

  gk_csr_CreateIndex(mat, GK_CSR_ROW);
  return mat;
}

/**************************************************************************/
/*! Computes ||y-Ax||_2^2 */
/**************************************************************************/
double ComputeResidual(gk_csr_t *mat, double *x, double *y) {
  ssize_t i, j;
  int32_t nrows;
  double res, r;
  ssize_t *rowptr;
  int32_t *rowind;
  float *rowval;

  nrows = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  for (res = 0.0, i = 0; i < nrows; i++) {
    if (rowval) {
      for (r = 0, j = rowptr[i]; j < rowptr[i + 1]; j++)
        r += x[rowind[j]] * rowval[j];
    } else {
      for (r = 0, j = rowptr[i]; j < rowptr[i + 1]; j++)
        r += x[rowind[j]];
    }
    res += (r - y[i]) * (r - y[i]);
  }
  return res;
}

/**************************************************************************/
/*! Computes the average predicted scores of the zero entries */
/**************************************************************************/
double ComputeAvgZeroScore(gk_csr_t *mat, double *x, double *y, int32_t ntop) {
  ssize_t i, j;
  int32_t nrows, nscores;
  double r;
  ssize_t *rowptr;
  int32_t *rowind;
  float *rowval;
  float *scores;

  nrows = mat->nrows;
  rowptr = mat->rowptr;
  rowind = mat->rowind;
  rowval = mat->rowval;

  scores = gk_fmalloc(nrows, "scores");

  for (nscores = 0, i = 0; i < nrows; i++) {
    if (y[i] > 0)
      continue;

    if (rowval) {
      for (r = 0, j = rowptr[i]; j < rowptr[i + 1]; j++)
        r += x[rowind[j]] * rowval[j];
    } else {
      for (r = 0, j = rowptr[i]; j < rowptr[i + 1]; j++)
        r += x[rowind[j]];
    }
    scores[nscores++] = r;
  }

  gk_fsortd(nscores, scores);
  ntop = gk_min(ntop, nscores);
  r = gk_fsum(ntop, scores, 1);
  gk_free((void **)&scores, LTERM);

  return (ntop == 0 ? 0 : r / ntop);
}
