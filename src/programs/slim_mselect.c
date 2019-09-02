/*!
\file
\brief The model selection stand-alone program for SLIM

\date    Started 3/11/2015
\author  George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\copyright 2019 Regents of the University of Minnesota
*/

#include "slimbin.h"

/*************************************************************************/
/*! The main entry for the learning */
/*************************************************************************/
int main(int argc, char *argv[]) {
  ssize_t zI;
  int32_t iU, iL, iR, rstatus, nrcmds, nhits[3], ntrue[2];
  int32_t nvalid, nvalid_head, nvalid_tail;
  float all_hr, head_hr, tail_hr;
  int is_tail_u, is_head_u;
  size_t nlines;
  params_t *params;
  gk_csr_t *trnmat, *tstmat;
  slim_t *model = NULL, *imodel = NULL;
  int32_t ioptions[SLIM_NOPTIONS];
  double doptions[SLIM_NOPTIONS];
  char **lines;
  int32_t *rids, *rmarker, *fmarker;
  float *rscores, hr[3], arhr, larhr, baseline;
  double timer;
  gk_csr_t *smat;

  /* parse command line */
  params = parse_cmdline(argc, argv);

  /* read the training/testing data */
  trnmat = gk_csr_Read(params->trnfile, params->ifmt, params->readvals, 0);
  tstmat = gk_csr_Read(params->tstfile, params->ifmt, params->readvals, 0);

  // trnmat = gk_csr_Read(params->trnfile, GK_CSR_FMT_CSR, 1, 1);
  // tstmat = gk_csr_Read(params->tstfile, GK_CSR_FMT_CSR, 1, 1);

  /* read the regularization sets as an array of strings */
  lines = gk_readfile(params->l12file, &nlines);

  /* free any user-supplied ratings if set to be ignored */
  if (params->binarize) {
    gk_free((void **)&trnmat->rowval, LTERM);
    gk_free((void **)&tstmat->rowval, LTERM);
  }

  if (trnmat->ncols < tstmat->ncols) {
    trnmat->ncols = tstmat->ncols;
  }

  printf(
      "------------------------------------------------------------------\n");
  printf("SLIM, version %s\n", SLIM_VERSION);
  printf(
      "------------------------------------------------------------------\n");
  printf("  trnfile: %s, nrows: %d, ncols: %d, nnz: %zd\n", params->trnfile,
         trnmat->nrows, trnmat->ncols, trnmat->rowptr[trnmat->nrows]);
  printf("  tstfile: %s, nrows: %d, ncols: %d, nnz: %zd\n", params->tstfile,
         tstmat->nrows, tstmat->ncols, tstmat->rowptr[tstmat->nrows]);
  printf("  optTol: %.2le, niters: %d\n", params->optTol, params->niters);
  printf("  binarize: %d, nnbrs: %d, nthreads: %d, dbglvl: %d\n",
         params->binarize, params->nnbrs, params->nthreads, params->dbglvl);
  printf("  simtype: %s\n", slim_simtypenames[params->simtype]);

  printf("\nEstimating & evaluating models...\n\n");

  /* set the defaults that do not change */
  SLIM_iSetDefaults(ioptions);
  SLIM_dSetDefaults(doptions);

  ioptions[SLIM_OPTION_DBGLVL] = params->dbglvl;
  ioptions[SLIM_OPTION_NNBRS] = params->nnbrs;
  ioptions[SLIM_OPTION_SIMTYPE] = params->simtype;
  ioptions[SLIM_OPTION_NTHREADS] = params->nthreads;
  ioptions[SLIM_OPTION_ALGO] = params->algo;
  ioptions[SLIM_OPTION_MAXNITERS] = params->niters;

  doptions[SLIM_OPTION_OPTTOL] = params->optTol;

  //  trnmat->ncols = 107397;
  /* allocate memory for the prediction arrays */
  rids = gk_i32malloc(params->nrcmds, "rids");
  rscores = gk_fmalloc(params->nrcmds, "rscores");
  rmarker = gk_i32malloc(trnmat->ncols, "rmarker");
  fmarker = SLIM_DetermineHeadAndTail(trnmat->nrows,
                                      gk_max(trnmat->ncols, tstmat->ncols),
                                      trnmat->rowptr, trnmat->rowind);

  double bestl1 = -1;
  double bestl2 = -1;
  double bestscore = 0.0;

  /* go over each set of l1/l2 values */
  for (iL = 0; iL < nlines; iL++) {
    sscanf(lines[iL], "%lf %lf", &doptions[SLIM_OPTION_L1R],
           &doptions[SLIM_OPTION_L2R]);

    /* learning */
    gk_clearwctimer(timer);
    gk_startwctimer(timer);
    imodel = model;
    model = SLIM_Learn(trnmat->nrows, trnmat->rowptr, trnmat->rowind,
                       trnmat->rowval, ioptions, doptions, imodel, &rstatus);
    gk_stopwctimer(timer);
    char model_file[100] = "";
    sprintf(model_file, "%s.model", lines[iL]);
    gk_csr_Write(model, model_file, params->ifmt, 1, 0);
    SLIM_FreeModel(&imodel);
    printf("Done!\n");
    if (rstatus != SLIM_OK) {
      printf("ERROR: Something went wrong with model estimation [%.3le %.3le]: "
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
      nrcmds = SLIM_GetTopN(
          model, trnmat->rowptr[iU + 1] - trnmat->rowptr[iU],
          trnmat->rowind + trnmat->rowptr[iU],
          (trnmat->rowval ? trnmat->rowval + trnmat->rowptr[iU] : NULL),
          ioptions, params->nrcmds, rids, rscores);

      nvalid += (nrcmds != SLIM_ERROR ? 1 : 0);
      is_tail_u = is_head_u = 0;
      larhr = baseline = 0.0;
      // initialize count of head and tail items
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
          // mark whether head or tail item was hit
          nhits[fmarker[rids[iR]]]++;
          // mark that an item was hit
          nhits[2]++;
          larhr += 1.0 / (1.0 + iR);
        }
      }

      // head hit rate in test data
      hr[0] += (nhits[0] > 0 ? 1.0 * nhits[0] / ntrue[0] : 0.0);
      // tail hit rate in test data
      hr[1] += (nhits[1] > 0 ? 1.0 * nhits[1] / ntrue[1] : 0.0);
      // overall hit rate in test data, (ratings hit in test / (number of
      // ratings in test))
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
    // Keep track of the best model
    if (all_hr > bestscore) {
      bestscore = all_hr;
      bestl1 = doptions[SLIM_OPTION_L1R];
      bestl2 = doptions[SLIM_OPTION_L2R];
    }
  }

  printf("\nDone.\n");
  printf(
      "------------------------------------------------------------------\n");

  printf("The selected hyperparameters are l1r: %.2f l2r: %.2f \n", bestl1,
         bestl2);

  printf(
      "------------------------------------------------------------------\n");

  /* clean up */
  SLIM_FreeModel(&model);
  gk_csr_Free(&trnmat);
  gk_csr_Free(&tstmat);
  gk_free((void **)&rids, &rscores, &rmarker, &fmarker, LTERM);
}
