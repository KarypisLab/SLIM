/*!
\file
\brief The user-callable API for SLIM

\date   Started 3/9/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/**************************************************************************/
/*! @brief  Wwrap a csr matrix from python to a gk_csr_t matrix
    @param  nrows      number of rows in the matrix
            rowptr     ""
            rowind     ""
            rowval     ""
            matrix_out handle to the output matrix
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_csr_wrapper(int32_t nrows, ssize_t *rowptr, int32_t *rowind,
                       float *rowval, slim_t **matrix_out) {
  gk_csr_t *mat = gk_csr_Create();
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
  *matrix_out = mat;
  return SLIM_OK;
}

/**************************************************************************/
/*! @brief  Save a gk_csr_t matrix from python
    @param  mathandle handle to the matrix
            fname     filename
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_csr_save(slim_t *mathandle, char *fname) {
  gk_csr_t *mat = (gk_csr_t *)mathandle;
  gk_csr_Write(mat, fname, GK_CSR_FMT_CSR, 1, 0);
  return SLIM_OK;
}

/**************************************************************************/
/*! @brief  load a gk_csr_t matrix to python
    @param  mathandle handle to the matrix
            fname     filename
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_csr_load(slim_t **mathandle, char *fname) {
  gk_csr_t *mat = gk_csr_Read(fname, GK_CSR_FMT_CSR, 1, 0);
  *mathandle = mat;
  return SLIM_OK;
}

/**************************************************************************/
/*! @brief  free a gk_csr_t matrix from python
    @param  mathandle handle to the matrix
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_csr_free(slim_t *mathandle) {
  gk_csr_t *mat = (gk_csr_t *)mathandle;
  gk_csr_Free(&mat);
  return SLIM_OK;
}

/**************************************************************************/
/*! @brief  get the statistics (nnz) of the csr model
    @param  mathandle handle to the matrix
            nnz       number of non-zeros in the model
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_csr_stat(slim_t *mathandle, int32_t *nnz) {
  gk_csr_t *mat = (gk_csr_t *)mathandle;
  *nnz = mat->rowptr[mat->nrows];
  return SLIM_OK;
}


/**************************************************************************/
/*! @brief  export the gk_csr matrix to a scipy csr matrix
    @param  mathandle handle to the matrix
            indptr    index pointer of the scipy csr matrix
            indices   index of the scipy csr matrix
            data      data of the scipy csr matrix
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_csr_export(slim_t *mathandle, int32_t *indptr, int32_t *indices, float *data) {
  int32_t nrows, nnz;

  gk_csr_t *mat = (gk_csr_t *)mathandle;
  nrows = mat->nrows;
  nnz = mat->rowptr[mat->nrows];

  for (int i = 0; i < nrows + 1; i++) {
    indptr[i] = mat->rowptr[i];
  }

  for (int i = 0; i < nnz; i++) {
    indices[i] = mat->rowind[i];
  }

  if (mat->rowval) {
    for (int i = 0; i < nnz; i++) {
      data[i] = mat->rowval[i];
    }
  }

  return SLIM_OK;
}

/**************************************************************************/
/*! @brief  estimate a slim model and return the model handle to python
    @param  trnhandle  handle to the training matrix
            ioptions   integer training options
            doptions   float training options
            model_out  handle to the output model(matrix)
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_SLIM_Learn(slim_t *trnhandle, int32_t *ioptions, double *doptions,
                      slim_t **model_out) {
  params_t params;
  gk_csr_t *tmat, *smat, *mat;
  slim_t *imodel = NULL;

  mat = (gk_csr_t *)trnhandle;

  /* setup params */
  memset((void *)&params, 0, sizeof(params_t));

  params.nthreads = GETOPTION(ioptions, SLIM_OPTION_NTHREADS, 1);
  params.nnbrs = GETOPTION(ioptions, SLIM_OPTION_NNBRS, 0);
  params.simtype = GETOPTION(ioptions, SLIM_OPTION_SIMTYPE, SLIM_SIMTYPE_COS);
  params.dbglvl = GETOPTION(ioptions, SLIM_OPTION_DBGLVL, 0);
  params.algo = GETOPTION(ioptions, SLIM_OPTION_ALGO, SLIM_ALGO_CD);
  params.ordered = GETOPTION(ioptions, SLIM_OPTION_ORDERED, 0);
  params.maxniters = GETOPTION(ioptions, SLIM_OPTION_MAXNITERS, 10000);

  params.l1r = GETOPTION(doptions, SLIM_OPTION_L1R, 1.0);
  params.l2r = GETOPTION(doptions, SLIM_OPTION_L2R, 1.0);
  params.optTol = GETOPTION(doptions, SLIM_OPTION_OPTTOL, 1e-7);

  params.mtype = SLIM_MTYPE_SLIM;
  if (params.nnbrs > 0 && params.ordered == 0)
    params.mtype = SLIM_MTYPE_FSLIM;
  if (params.nnbrs > 0 && params.ordered == 1)
    params.mtype = SLIM_MTYPE_OFSLIM;
  if (params.nnbrs == 0 && params.ordered == 1)
    params.mtype = SLIM_MTYPE_OSLIM;

  IFSET(params.dbglvl, SLIM_DBG_INFO, PrintParams(&params));

  InitTimers(&params);
  gk_startwctimer(params.TotalTmr);

  /* setup the training data */
  gk_startwctimer(params.SetupTmr);
  tmat = CreateTrainingMatrix(&params, mat->nrows, mat->rowptr, mat->rowind,
                              mat->rowval);
  gk_stopwctimer(params.SetupTmr);

  /* estimate the model */
  gk_startwctimer(params.LearnTmr);
  switch (params.algo) {
  case SLIM_ALGO_ADMM:
    smat = EstimateModelADMM(&params, tmat, (gk_csr_t *)imodel);
    break;
  case SLIM_ALGO_CD:
    smat = EstimateModelCD(&params, tmat, (gk_csr_t *)imodel);
    break;
  default:
    printf("Algorithm not supported.\n");
    exit(0);
  }
  gk_stopwctimer(params.LearnTmr);

  gk_stopwctimer(params.TotalTmr);
  IFSET(params.dbglvl, SLIM_DBG_TIME, PrintTimers(&params));

  /* free the data */
  gk_csr_Free(&tmat);
  *model_out = smat;

  return SLIM_OK;
}

/**************************************************************************/
/*! @brief  train and validate models with various combination of l1 and l2
    @param  trnhandle  handle to the training matrix
            tsthandle  handle to the test matrix
            ioptions   integer training options
            doptions   float training options
            arrayl1    array of l1s
            arrayl2    array of l2s
            nl1        length of arrayl1
            nl2        length of arrayl2
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_SLIM_Mselect(slim_t *trnhandle, slim_t *tsthandle, int32_t *ioptions,
                        double *doptions, double *arrayl1, double *arrayl2,
                        int32_t nl1, int32_t nl2, double *bestl1HR,
                        double *bestl2HR, double *bestHRHR, double *bestARHR,
                        double *bestl1AR, double *bestl2AR, double *bestHRAR,
                        double *bestARAR) {
  ssize_t zI;
  int32_t iU, iL1, iL2, iR, rstatus, nrcmds, nhits[3], ntrue[2];
  int32_t nvalid, nvalid_head, nvalid_tail;
  float all_hr, head_hr, tail_hr;
  int is_tail_u, is_head_u;
  params_t params;
  gk_csr_t *trnmat, *tstmat, *trn, *tst;
  slim_t *model = NULL, *imodel = NULL;
  int32_t *rids, *rmarker, *fmarker;
  float *rscores, hr[3], arhr, larhr, baseline;
  double timer;
  gk_csr_t *smat;

  trn = (gk_csr_t *)trnhandle;
  tst = (gk_csr_t *)tsthandle;

  /* setup params */
  memset((void *)&params, 0, sizeof(params_t));

  params.nthreads = GETOPTION(ioptions, SLIM_OPTION_NTHREADS, 1);
  params.nnbrs = GETOPTION(ioptions, SLIM_OPTION_NNBRS, 0);
  params.simtype = GETOPTION(ioptions, SLIM_OPTION_SIMTYPE, SLIM_SIMTYPE_COS);
  params.dbglvl = GETOPTION(ioptions, SLIM_OPTION_DBGLVL, 0);
  params.algo = GETOPTION(ioptions, SLIM_OPTION_ALGO, SLIM_ALGO_CD);
  params.ordered = GETOPTION(ioptions, SLIM_OPTION_ORDERED, 0);
  params.nrcmds = GETOPTION(ioptions, SLIM_OPTION_NRCMDS, 10);
  params.maxniters = GETOPTION(ioptions, SLIM_OPTION_MAXNITERS, 10000);

  params.optTol = GETOPTION(doptions, SLIM_OPTION_OPTTOL, 1e-7);

  /* setup the training data */
  trnmat = CreateTrainingMatrix(&params, trn->nrows, trn->rowptr, trn->rowind,
                                trn->rowval);
  tstmat = CreateTrainingMatrix(&params, tst->nrows, tst->rowptr, tst->rowind,
                                tst->rowval);

  if (trnmat->ncols < tstmat->ncols) {
    trnmat->ncols = tstmat->ncols;
  }

  printf(
      "------------------------------------------------------------------\n");
  printf("SLIM, version %s\n", SLIM_VERSION);
  printf(
      "------------------------------------------------------------------\n");
  printf("  trn matrix, nrows: %d, ncols: %d, nnz: %zd\n", trnmat->nrows,
         trnmat->ncols, trnmat->rowptr[trnmat->nrows]);
  printf("  tst matrix, nrows: %d, ncols: %d, nnz: %zd\n", tstmat->nrows,
         tstmat->ncols, tstmat->rowptr[tstmat->nrows]);
  printf("  optTol: %.2le, niters: %d\n", params.optTol, params.maxniters);
  printf("  nnbrs: %d, nthreads: %d, ordered: %d, dbglvl: %d\n", params.nnbrs,
         params.nthreads, params.ordered, params.dbglvl);
  printf("  simtype: %s\n", slim_simtypenames[params.simtype]);

  printf("\nEstimating & evaluating models...\n\n");

  /* allocate memory for the prediction arrays */
  rids = gk_i32malloc(params.nrcmds, "rids");
  rscores = gk_fmalloc(params.nrcmds, "rscores");
  rmarker = gk_i32malloc(trnmat->ncols, "rmarker");
  fmarker = SLIM_DetermineHeadAndTail(trnmat->nrows,
                                      gk_max(trnmat->ncols, tstmat->ncols),
                                      trnmat->rowptr, trnmat->rowind);

  *bestHRHR = 0., *bestARHR = 0., *bestHRAR = 0., *bestARAR = 0.;
  /* go over each set of l1/l2 values */
  for (iL1 = 0; iL1 < nl1; iL1++) {
    for (iL2 = 0; iL2 < nl2; iL2++) {
      doptions[SLIM_OPTION_L1R] = arrayl1[iL1];
      doptions[SLIM_OPTION_L2R] = arrayl2[iL2];

      /* learning */
      gk_clearwctimer(timer);
      gk_startwctimer(timer);
      imodel = model;
      model = SLIM_Learn(trnmat->nrows, trnmat->rowptr, trnmat->rowind,
                         trnmat->rowval, ioptions, doptions, imodel, &rstatus);
      gk_stopwctimer(timer);
      SLIM_FreeModel(&imodel);
      printf("eftase\n");
      if (rstatus != SLIM_OK) {
        printf(
            "ERROR: Something went wrong with model estimation [%.3le %.3le]: "
            "rstatus: %" PRId32 "\n",
            doptions[SLIM_OPTION_L1R], doptions[SLIM_OPTION_L2R], rstatus);
        continue;
      }

      /* test the model */
      gk_i32set(trnmat->ncols, -1, rmarker);
      hr[0] = hr[1] = hr[2] = 0.0;
      arhr = 0.0;
      nvalid = nvalid_head = nvalid_tail = 0;

      for (iU = 0; iU < trnmat->nrows; iU++) {
        if (tstmat->rowptr[iU + 1] - tstmat->rowptr[iU] < 1) continue;
        nrcmds = SLIM_GetTopN(
            model, trnmat->rowptr[iU + 1] - trnmat->rowptr[iU],
            trnmat->rowind + trnmat->rowptr[iU],
            (trnmat->rowval ? trnmat->rowval + trnmat->rowptr[iU] : NULL),
            ioptions, params.nrcmds, rids, rscores);

        nvalid += (nrcmds != SLIM_ERROR ? 1 : 0);
        is_tail_u = is_head_u = 0;
        larhr = baseline = 0.0;
        ntrue[0] = ntrue[1] = 0;

        for (zI = tstmat->rowptr[iU]; zI < tstmat->rowptr[iU + 1]; zI++) {
          rmarker[tstmat->rowind[zI]] = iU;
          ntrue[fmarker[tstmat->rowind[zI]]]++;
          if (fmarker[tstmat->rowind[zI]]) {
            // tail
            is_tail_u = 1;
          } else {
            // head
            is_head_u = 1;
          }
          baseline += 1.0 / (1.0 + zI - tstmat->rowptr[iU]);
        }

        if (is_tail_u) {
          nvalid_tail++;
        }

        if (is_head_u) {
          nvalid_head++;
        }

        nhits[0] = nhits[1] = nhits[2] = 0;
        for (iR = 0; iR < nrcmds; iR++) {
          if (rmarker[rids[iR]] == iU) {
            nhits[fmarker[rids[iR]]]++;
            nhits[2]++;
            larhr += 1.0 / (1.0 + iR);
          }
        }

        hr[0] += (nhits[0] > 0 ? 1.0 * nhits[0] / ntrue[0] : 0.0);
        hr[1] += (nhits[1] > 0 ? 1.0 * nhits[1] / ntrue[1] : 0.0);
        hr[2] += 1.0 * nhits[2] / (tstmat->rowptr[iU + 1] - tstmat->rowptr[iU]);
        arhr += larhr / baseline;
      }

      all_hr = nvalid > 0 ? hr[2] / nvalid : 0;
      head_hr = nvalid_head > 0 ? hr[0] / nvalid_head : 0;
      tail_hr = nvalid_tail > 0 ? hr[1] / nvalid_tail : 0;
      arhr = nvalid > 0 ? arhr / nvalid : 0;

      smat = (gk_csr_t *)model;
      printf("\nnvalid: %d nvalid_head: %d nvalid_tail: %d", nvalid, nvalid_head,
             nvalid_tail);
      printf("\nl1r: %.2le l2r: %.2le nnz: %7zd"
             " hr: %.4f hr_head: %.4f hr_tail: %.4f arhr: %.4f time: %.2lf\n",
             doptions[SLIM_OPTION_L1R], doptions[SLIM_OPTION_L2R],
             smat->rowptr[smat->nrows], all_hr, head_hr, tail_hr, arhr,
             gk_getwctimer(timer));

      if (nvalid < 1) {
        *bestl1HR = doptions[SLIM_OPTION_L1R];
        *bestl2HR = doptions[SLIM_OPTION_L2R];
        return SLIM_ERROR;
      }

      /* model selection in hr */
      if (all_hr > *bestHRHR) {
        *bestHRHR = all_hr;
        *bestARHR = arhr;
        *bestl1HR = doptions[SLIM_OPTION_L1R];
        *bestl2HR = doptions[SLIM_OPTION_L2R];
      }

      /* model selection in ar */
      if (arhr > *bestARAR) {
        *bestHRAR = all_hr;
        *bestARAR = arhr;
        *bestl1AR = doptions[SLIM_OPTION_L1R];
        *bestl2AR = doptions[SLIM_OPTION_L2R];
      }
    }
  }

  printf("\nDone.\n");
  printf(
      "------------------------------------------------------------------\n");

  /* clean up */
  SLIM_FreeModel(&model);
  gk_csr_Free(&trnmat);
  gk_csr_Free(&tstmat);
  gk_free((void **)&rids, &rscores, &rmarker, &fmarker, LTERM);

  return SLIM_OK;
}

int32_t Py_SLIM_GetTopN(slim_t *model, int32_t nratings, int32_t *itemids,
                        float *ratings, int32_t nrcmds, int32_t *rids,
                        float *rscores, int32_t dbglvl) {
  params_t params;
  gk_csr_t *smat;

  /* setup params */
  memset((void *)&params, 0, sizeof(params_t));

  params.dbglvl = dbglvl;

  InitTimers(&params);

  /* get the model in the internal form */
  smat = (gk_csr_t *)model;

  /* get the recommendations */
  gk_startwctimer(params.TotalTmr);
  nrcmds = GetRecommendations(&params, smat, nratings, itemids, ratings, nrcmds,
                              rids, rscores);
  gk_stopwctimer(params.TotalTmr);

  if (nrcmds < 0)
    return SLIM_ERROR;
  else
    return nrcmds;
}

int32_t Py_SLIM_GetTopN_1vsk(slim_t *model, int32_t nratings, int32_t *itemids,
                        float *ratings, int32_t nrcmds, int32_t *rids,
                        float *rscores, int32_t nnegs, int32_t *negitems, 
                        int32_t dbglvl) {
  params_t params;
  gk_csr_t *smat;

  /* setup params */
  memset((void *)&params, 0, sizeof(params_t));

  params.dbglvl = dbglvl;

  InitTimers(&params);

  /* get the model in the internal form */
  smat = (gk_csr_t *)model;

  /* get the recommendations */
  gk_startwctimer(params.TotalTmr);
  nrcmds = GetRec_1vsk(&params, smat, nratings, itemids, ratings, nrcmds,
                              rids, rscores, nnegs, negitems);
  gk_stopwctimer(params.TotalTmr);

  if (nrcmds < 0)
    return SLIM_ERROR;
  else
    return nrcmds;
}

/**************************************************************************/
/*! @brief  predict topn lists
    @param  nrcmds      number of items to be recommended
            nnegs       number of negative items
            slimhandle  handle to the training matrix
            trnhandle   integer training options
            negitems    pointer to the negative items
            output      pointer to the output lists
            scores      pointer to the output scores
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_SLIM_Predict_1vsk(int32_t nrcmds, int32_t nnegs, slim_t *slimhandle, slim_t *trnhandle,
                        int32_t *negitems, int32_t *output, float *scores) {
  int32_t iU, iR, n, nvalid = 0;
  int32_t *rids;
  float *rscores;
  gk_csr_t *model, *trnmat;

  model = (gk_csr_t *)slimhandle;
  trnmat = (gk_csr_t *)trnhandle;

  rids = gk_i32malloc(nrcmds, "rids");
  rscores = gk_fmalloc(nrcmds, "rscores");

  for (iU = 0; iU < trnmat->nrows; iU++) {
    n = Py_SLIM_GetTopN_1vsk(
        model, trnmat->rowptr[iU + 1] - trnmat->rowptr[iU],
        trnmat->rowind + trnmat->rowptr[iU],
        (trnmat->rowval ? trnmat->rowval + trnmat->rowptr[iU] : NULL), nrcmds,
        rids, rscores, nnegs, negitems + iU * nnegs, 0);

    if (n != SLIM_ERROR) {
      for (iR = 0; iR < n; iR++) {
        output[iU * nrcmds + iR] = rids[iR];
        scores[iU * nrcmds + iR] = rscores[iR];
        // printf("id: %d", rids[iR]);
      }
      nvalid += 1;
      // printf("---\n");
    }
  }
  if (nvalid < 1) {
    return SLIM_ERROR;
  } else {
    return SLIM_OK;
  }
}

/**************************************************************************/
/*! @brief  predict topn lists
    @param  nrcmds      number of items to be recommended
            slimhandle  handle to the training matrix
            trnhandle   integer training options
            output      pointer to the output lists
            scores      pointer to the output scores
    @return a flag indicating whether the function succeed
*/
/**************************************************************************/
int32_t Py_SLIM_Predict(int32_t nrcmds, slim_t *slimhandle, slim_t *trnhandle,
                        int32_t *output, float *scores) {
  int32_t iU, iR, n, nvalid = 0;
  int32_t *rids;
  float *rscores;
  gk_csr_t *model, *trnmat;

  model = (gk_csr_t *)slimhandle;
  trnmat = (gk_csr_t *)trnhandle;

  rids = gk_i32malloc(nrcmds, "rids");
  rscores = gk_fmalloc(nrcmds, "rscores");

  for (iU = 0; iU < trnmat->nrows; iU++) {
    n = Py_SLIM_GetTopN(
        model, trnmat->rowptr[iU + 1] - trnmat->rowptr[iU],
        trnmat->rowind + trnmat->rowptr[iU],
        (trnmat->rowval ? trnmat->rowval + trnmat->rowptr[iU] : NULL), nrcmds,
        rids, rscores, 0);

    if (n != SLIM_ERROR) {
      for (iR = 0; iR < n; iR++) {
        output[iU * nrcmds + iR] = rids[iR];
        scores[iU * nrcmds + iR] = rscores[iR];
      }
      nvalid += 1;
    }
  }
  if (nvalid < 1) {
    return SLIM_ERROR;
  } else {
    return SLIM_OK;
  }
}
