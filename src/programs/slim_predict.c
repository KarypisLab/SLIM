/*!
\file
\brief The stand-alone top-N recommendation program for SLIM

\date    Started 3/13/2015
\author  Xia & George
\copyright 2011-2015 Regents of the University of Minnesota
*/

#include "slimbin.h"

/*************************************************************************/
/*! The main entry for the prediction */
/*************************************************************************/
int main(int argc, char *argv[]) {
  ssize_t zI;
  int32_t iU, iR, nrcmds, nhits[3], ntrue[2];
  int32_t nvalid, nvalid_head, nvalid_tail;
  float all_hr, head_hr, tail_hr;
  int is_tail_u, is_head_u;
  int32_t *rids, *rmarker, *fmarker;
  float *rscores, hr[3], arhr, larhr, baseline;
  params_t *params;
  gk_csr_t *oldmat, *tstmat = NULL, *model;
  int32_t ioptions[SLIM_NOPTIONS];
  FILE *fpout = NULL;

  /* parse command line */
  params = parse_cmdline(argc, argv);

  /* read the various data files */
  //  model  = (gk_csr_t *)SLIM_ReadModel(params->mdlfile);
  model = gk_csr_Read(params->mdlfile, params->ifmt, params->readvals, 0);
  oldmat = gk_csr_Read(params->trnfile, params->ifmt, params->readvals, 0);
  if (params->tstfile)
    tstmat = gk_csr_Read(params->tstfile, params->ifmt, params->readvals, 0);

  printf(
      "------------------------------------------------------------------\n");
  printf("SLIM, version %s\n", SLIM_VERSION);
  printf(
      "------------------------------------------------------------------\n");
  printf("  mdlfile: %s, nrows: %d, ncols: %d, nnz: %zd\n", params->mdlfile,
         model->nrows, model->ncols, model->rowptr[model->nrows]);
  printf("  oldfile: %s, nrows: %d, ncols: %d, nnz: %zd\n", params->trnfile,
         oldmat->nrows, oldmat->ncols, oldmat->rowptr[oldmat->nrows]);
  if (tstmat)
    printf("  tstfile: %s, nrows: %d, ncols: %d, nnz: %zd\n", params->tstfile,
           tstmat->nrows, tstmat->ncols, tstmat->rowptr[tstmat->nrows]);
  if (params->outfile)
    printf("  outfile: %s\n",
           (params->outfile ? params->outfile : "No output"));
  printf("  binarize: %d, nrcmds: %d, dbglvl: %d\n", params->binarize,
         params->nrcmds, params->dbglvl);
  printf("\nMaking predictions...\n");

  if (tstmat && oldmat->nrows != tstmat->nrows)
    errexit("The number of rows in the old and test files do not match.\n");

  /* free any user-supplied ratings if set to be ignored */
  if (params->binarize) {
    gk_free((void **)&oldmat->rowval, LTERM);
    if (tstmat)
      gk_free((void **)&tstmat->rowval, LTERM);
  }

  SLIM_iSetDefaults(ioptions);
  ioptions[SLIM_OPTION_DBGLVL] = params->dbglvl;

  /* predict for each row in oldmat */
  if (params->outfile)
    fpout = gk_fopen(params->outfile, "w", "outfile");

  rids = gk_i32malloc(params->nrcmds, "rids");
  rscores = gk_fmalloc(params->nrcmds, "rscores");
  rmarker = (tstmat ? gk_i32smalloc(model->ncols, -1, "rmarker") : NULL);
  // get head and tail columns, mark 0 for head items and 1 for items in tail
  fmarker = (tstmat ? SLIM_DetermineHeadAndTail(
                          oldmat->nrows, gk_max(oldmat->ncols, tstmat->ncols),
                          oldmat->rowptr, oldmat->rowind)
                    : NULL);

  hr[0] = hr[1] = hr[2] = 0.0;
  arhr = 0.0;
  nvalid = nvalid_head = nvalid_tail = 0;

  for (iU = 0; iU < oldmat->nrows; iU++) {
    nrcmds = SLIM_GetTopN(
        model, oldmat->rowptr[iU + 1] - oldmat->rowptr[iU],
        oldmat->rowind + oldmat->rowptr[iU],
        (oldmat->rowval ? oldmat->rowval + oldmat->rowptr[iU] : NULL), ioptions,
        params->nrcmds, rids, rscores);

    nvalid += (nrcmds != SLIM_ERROR ? 1 : 0);
    is_tail_u = is_head_u = 0;
    /* save the recommendations */
    if (fpout) {
      if (nrcmds != SLIM_ERROR) {
        for (iR = 0; iR < nrcmds; iR++)
          fprintf(fpout, " %" PRId32 " %f", rids[iR], rscores[iR]);
        fprintf(fpout, "\n");
      } else {
        fprintf(fpout, "-1\n");
      }
    }

    /* evaluate the recommendations against the test set */
    if (tstmat) {
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
  }

  all_hr = nvalid > 0 ? hr[2] / nvalid : 0;
  head_hr = nvalid_head > 0 ? hr[0] / nvalid_head : 0;
  tail_hr = nvalid_tail > 0 ? hr[1] / nvalid_tail : 0;
  arhr = nvalid > 0 ? arhr / nvalid : 0;

  if (fpout) {
    gk_fclose(fpout);
  }
  printf("\nnvalid: %d nvalid_head: %d nvalid_tail: %d", nvalid, nvalid_head,
         nvalid_tail);
  printf("\nhr: %.4f hr_head: %.4f hr_tail: %.4f arhr: %.4f\n", all_hr, head_hr,
         tail_hr, arhr);
  printf(
      "------------------------------------------------------------------\n");

  /* clean up */
  gk_free((void **)&rids, &rscores, &rmarker, &fmarker, LTERM);
  SLIM_FreeModel((slim_t **)&model);
  gk_csr_Free(&oldmat);
  if (tstmat)
    gk_csr_Free(&tstmat);
}
