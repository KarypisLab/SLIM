/**
 * @file
 * @brief The user-callable API for SLIM
 * @date   Started 3/9/2015
 * @author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
 * @author Copyright 2019, Regents of the University of Minnesota
 */

#include "slimlib.h"

/** \addtogroup slimapi Functions for using the SLIM library
 *  This is the set of methods that can used from SLIM library.
 *
 *  @{
 */

/**
 * @brief Entry point for the model estimation routine. It uses the sparse
 * matrix in CSR format.
 *
 * @param nrows number of rows in the matrix
 * @param rowptr [rowptr[i], rowptr[i+1]) points to the indices in rowind and
 * rowval
 * @param rowind contains the column indices of the non-zero elements in matrix.
 * @param rowval contains the values of the non-zero elements in matrix.
 * @param ioptions integer or boolean options to the estimation routine.
 * @param doptions double options to the estimation routine.
 * @param imodel if not null then it is pointer to model that will be used for
 * initialization.
 * @param r_status set to 1 on success
 * @return sparse representation of model in CSR format.
 */
slim_t *SLIM_Learn(int32_t nrows, ssize_t *rowptr, int32_t *rowind,
                   float *rowval, int32_t *ioptions, double *doptions,
                   slim_t *imodel, int32_t *r_status) {
  params_t params;
  gk_csr_t *tmat, *smat;

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
  tmat = CreateTrainingMatrix(&params, nrows, rowptr, rowind, rowval);
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

  *r_status = SLIM_OK;

  return (slim_t *)smat;
}

/**
 * @brief get Top-N recommendations given a historical rating profile
 *
 * @param model the SLIM model matrix
 * @param nratings number of ratings in the historical profile
 * @param itemids ids of rated items in the historical profile
 * @param ratings ratings of items in the historical profile
 * @param ioptions integer options passed to the routine
 * @param nrcmds N in Top-N, i.e., size of recommendation list
 * @param rids ids of items in the recommendation list
 * @param rscores predicted ratings items in the recommendation list
 * @return  size of recommendation list on success else a value < 0
 */
int32_t SLIM_GetTopN(slim_t *model, int32_t nratings, int32_t *itemids,
                     float *ratings, int32_t *ioptions, int32_t nrcmds,
                     int32_t *rids, float *rscores) {
  params_t params;
  gk_csr_t *smat;

  /* setup params */
  memset((void *)&params, 0, sizeof(params_t));

  params.dbglvl = GETOPTION(ioptions, SLIM_OPTION_DBGLVL, 0);

  /* IFSET(params.dbglvl, SLIM_DBG_INFO, PrintParams(&params)); */

  InitTimers(&params);

  /* get the model in the internal form */
  smat = (gk_csr_t *)model;

  /* get the recommendations */
  gk_startwctimer(params.TotalTmr);
  nrcmds = GetRecommendations(&params, smat, nratings, itemids, ratings, nrcmds,
                              rids, rscores);
  gk_stopwctimer(params.TotalTmr);

  /* IFSET(params.dbglvl, SLIM_DBG_TIME, PrintTimers(&params)); */

  if (nrcmds < 0)
    return SLIM_ERROR;
  else
    return nrcmds;
}

/**
 * @brief Sets the default value (-1) for passed options
 *
 * @param options the integer array having option values
 * @return 1 on success
 */
int32_t SLIM_iSetDefaults(int32_t *options) {
  gk_i32set(SLIM_NOPTIONS, -1, options);

  return SLIM_OK;
}

/**
 * @brief Sets the default value (-1) for passed options
 *
 * @param options the double array having option values
 * @return 1 on success
 */
int32_t SLIM_dSetDefaults(double *options) {
  gk_dset(SLIM_NOPTIONS, -1, options);

  return SLIM_OK;
}

/**
 * @brief Writes the model to a supplied file.
 *
 * @param model the SLIM model matrix
 * @param filename the name of the file to write model to in CSR format
 * @return 1 on success
 */
int32_t SLIM_WriteModel(slim_t *model, char *filename) {
  gk_csr_Write((gk_csr_t *)model, filename, GK_CSR_FMT_BINROW, 1, 0);
  return SLIM_OK;
}

/**
 * @brief Reads the model from the passed file in CSR format. For example:
 *  \code{.c}
 *    slim_t *model = SLIM_ReadModel(input_model_filename);
 *  \endcode
 * @param filename the name of the file having model in CSR format
 * @return return the SLIM model sparse matrix in CSR format
 */
slim_t *SLIM_ReadModel(char *filename) {
  gk_csr_t *model;

  model = gk_csr_Read(filename, GK_CSR_FMT_BINROW, 1, 0);
  gk_csr_CreateIndex(model, GK_CSR_COL);

  return (slim_t *)model;
}

/**
 * @brief Frees the memory allocated for the SLIM model matrix. For example:
 *  \code{.c}
 *    slim_t *model = SLIM_ReadModel(input_model_filename);
 *    SLIM_FreeModel(&model);
 *  \endcode
 * @param r_model the SLIM model sparse matrix
 */
void SLIM_FreeModel(slim_t **r_model) { gk_csr_Free((gk_csr_t **)r_model); }

/* for internal use  */

/**
 * @brief  Returns an array marking as 0 the columns that belong to the head and
 * as 1 the columns that belong to the tail. The split is based on an 50-50
 * split (head: the most frequent items that correspond to the 50% of the
 * ratings).
 *
 */
int32_t *SLIM_DetermineHeadAndTail(int32_t nrows, int32_t ncols,
                                   ssize_t *rowptr, int32_t *rowind) {
  ssize_t zI, cnnz;
  int32_t iR, iC, *fmarker = NULL;
  gk_ikv_t *cand;

  fmarker = gk_i32smalloc(ncols, 1, "fmarker");
  cand = gk_ikvmalloc(ncols, "cand");

  for (iC = 0; iC < ncols; iC++) {
    cand[iC].key = 0;
    cand[iC].val = iC;
  }

  for (iR = 0; iR < nrows; iR++) {
    for (zI = rowptr[iR]; zI < rowptr[iR + 1]; zI++)
      cand[rowind[zI]].key++;
  }

  gk_ikvsortd(ncols, cand);
  cnnz = rowptr[nrows] / 2;

  for (iC = 0; iC < ncols && cnnz > 0; iC++) {
    fmarker[cand[iC].val] = 0;
    cnnz -= cand[iC].key;
  }

  gk_free((void **)&cand, LTERM);

  return fmarker;
}

/** @} */
/*************************************************************************/
/*! This function prints the various parameter fields */
/*************************************************************************/
void PrintParams(params_t *params) {
  printf(" Runtime parameters:\n");

  printf("   Model type: ");
  switch (params->mtype) {
  case SLIM_MTYPE_SLIM:
    printf("SLIM\n");
    break;
  case SLIM_MTYPE_FSLIM:
    printf("FSLIM (%" PRId32 " %s)\n", params->nnbrs,
           slim_simtypenames[params->simtype]);
    break;
  case SLIM_MTYPE_OSLIM:
    printf("OSLIM\n");
    break;
  case SLIM_MTYPE_OFSLIM:
    printf("OFSLIM (%" PRId32 " %s)\n", params->nnbrs,
           slim_simtypenames[params->simtype]);
    break;
  default:
    printf("Unknown!\n");
  }

  printf("   Optimization: l1r: %.2le, l2r: %.2le\n"
         "                 optTol: %.2le, maxniters: %" PRId32 "\n",
         params->l1r, params->l2r, params->optTol, params->maxniters);

  printf("   nthreads: %" PRId32 "\n", params->nthreads);

  printf("\n");
}
