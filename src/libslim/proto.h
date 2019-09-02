/*!
\file
\brief Function prototypes

\date   Started 3/9/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/


#ifndef __LIBSLIMPROTO_H__
#define __LIBSLIMPROTO_H__

/* api.c */
void PrintParams(params_t *params);

/* setup.c */
gk_csr_t *CreateTrainingMatrix(params_t *params, int32_t nrows, ssize_t *rowptr,
              int32_t *rowind, float *rowval);

/* cd.c */
int32_t CoordinateDescent(wspace_t *wspace, double *x, double *y, double *yhat, int32_t nacols);

/* estimate.c */
gk_csr_t *EstimateModelADMM(params_t *params, gk_csr_t *tmat, gk_csr_t *imat);
gk_csr_t *EstimateModelCD(params_t *params, gk_csr_t *tmat, gk_csr_t *imat);
gk_csr_t *SaveModel(ssize_t tnnz, int32_t nrows, int32_t ncols, int32_t *nnzs, gk_fkv_t **lists);
int myAprod(const int mode, const int m, const int n, const int nix, int *ix,
        double *x, double *y, void *UsrWrk);
double ComputeResidual(gk_csr_t *mat, double *x, double *y);
double ComputeAvgZeroScore(gk_csr_t *mat, double *x, double *y, int32_t ntop);

/* predict.c */
int32_t GetRecommendations(params_t *params, gk_csr_t *smat, int32_t nratings,
            int32_t *itemids, float *ratings, int32_t nrcmds, int32_t *rids,
            float *rscores);

/* timing.c */
void InitTimers(params_t *params);
void PrintTimers(params_t *params);

/* neighbors.c */
int32_t FindColumnNeighbors(params_t *params, wspace_t *wspace, gk_csr_t *mat,
            int32_t iC);

#endif
