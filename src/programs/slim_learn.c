/*!
\file
\brief The model estimation stand-alone program for SLIM

\date    Started 3/11/2015
\author  George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\copyright 2019 Regents of the University of Minnesota
*/

#include "slimbin.h"

/*************************************************************************/
/*! The main entry for the learning */
/*************************************************************************/
int main(int argc, char *argv[]) {
  int32_t rstatus;
  params_t *params;
  gk_csr_t *tmat, *imodel;
  slim_t *model;
  int32_t ioptions[SLIM_NOPTIONS];
  double doptions[SLIM_NOPTIONS];

  /* parse command line */
  params = parse_cmdline(argc, argv);

  /* read the training data */
  tmat = gk_csr_Read(params->trnfile, params->ifmt, params->readvals, 0);

  printf(
      "------------------------------------------------------------------\n");
  printf("SLIM, version %s\n", SLIM_VERSION);
  printf(
      "------------------------------------------------------------------\n");
  printf("  trnfile: %s, nrows: %d, ncols: %d, nnz: %zd\n", params->trnfile,
         tmat->nrows, tmat->ncols, tmat->rowptr[tmat->nrows]);
  printf("  l1r: %.2le, l2r: %.2le, optTol: %.2le, niters: %d\n", params->l1r,
         params->l2r, params->optTol, params->niters);
  printf("  binarize: %d, nnbrs: %d, nthreads: %d, dbglvl: %d\n",
         params->binarize, params->nnbrs, params->nthreads, params->dbglvl);
  printf("  simtype: %s, mdlfile: %s\n", slim_simtypenames[params->simtype],
         params->mdlfile);
  printf("\nEstimating model...\n");

  /* free any user-supplied ratings if set to be ignored */
  if (params->binarize)
    gk_free((void **)&tmat->rowval, LTERM);

  // read input model file if provided
  imodel = NULL;
  if (params->ipmdlfile) {
    imodel = gk_csr_Read(params->ipmdlfile, GK_CSR_FMT_CSR, 1, 0);
    gk_csr_CreateIndex(imodel, GK_CSR_COL);
    // check if number of columns same as that of input training matrix
    GKASSERT(imodel->nrows == tmat->ncols);
  }

  SLIM_iSetDefaults(ioptions);
  SLIM_dSetDefaults(doptions);

  ioptions[SLIM_OPTION_DBGLVL] = params->dbglvl;
  ioptions[SLIM_OPTION_NNBRS] = params->nnbrs;
  ioptions[SLIM_OPTION_SIMTYPE] = params->simtype;
  ioptions[SLIM_OPTION_ALGO] = params->algo;
  ioptions[SLIM_OPTION_NTHREADS] = params->nthreads;
  ioptions[SLIM_OPTION_MAXNITERS] = params->niters;

  doptions[SLIM_OPTION_L1R] = params->l1r;
  doptions[SLIM_OPTION_L2R] = params->l2r;
  doptions[SLIM_OPTION_OPTTOL] = params->optTol;

  /* learning */
  model = SLIM_Learn(tmat->nrows, tmat->rowptr, tmat->rowind, tmat->rowval,
                     ioptions, doptions, (slim_t *)imodel, &rstatus);

  if (rstatus != SLIM_OK) {
    printf(
        "ERROR: Something went wrong with model estimation: rstatus: %" PRId32
        "\n",
        rstatus);
  } else if (params->mdlfile) {
    gk_csr_Write(model, params->mdlfile, params->ifmt, 1, 0);
  }

  printf("\nDone.\n");
  printf(
      "------------------------------------------------------------------\n");

  /* clean up */
  SLIM_FreeModel(&model);
  gk_csr_Free(&tmat);
}
