/*!
\file
\brief Various data structures 

\date   Started 03/07/15
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/


#ifndef __LIBSTRUCT_H__
#define __LIBSTRUCT_H__



/**************************************************************/
/*! The parameters associated with learning */
/**************************************************************/
typedef struct {
  slim_mtype_et mtype;        /*!< type of model */

  /* user supplied parameters */
  double l1r;        /*!< the regularization parameter for L-1 norm */
  double l2r;        /*!< the regularization parameter for L-2 norm */
  double optTol;     /*!< optimality tolerance */
  int maxniters;     /*!< max number of iterations allowed in BCLS solver */
  int32_t nnbrs;     /*!< the # of neighbors in the FSLIM model */
  int32_t nthreads;  /*!< the number of threads to use */
  int32_t ordered;   /*!< turns on Seq[F]SLIM model */
  int32_t algo;      /*!< optimization algorithm to train the model */
  int32_t dbglvl;    /*!< level of debugging output */
  int32_t nrcmds;    /*!< the # of items to be recommended */

  slim_simtype_et simtype;   /*!< the type of similarity to use in FSLIM */       

  double TotalTmr, SetupTmr, LearnTmr, Aux1Tmr, Aux2Tmr, Aux3Tmr;

} params_t; 


/**************************************************************/
/*! The workspace structure used for CD. 
 */
/**************************************************************/
typedef struct 
{
  ssize_t *colptr;   /* the column view of the sparse matrix */
  int32_t *colind;   /* '' */
  float *colval;     /* '' */
  float *cnorms;     /* '' */

  gk_fkv_t *cdacols;    /* marks the allowed regressors (columns) */

  params_t *params;  /* hook to the run parameters */

  int32_t niters;    /* keeps track of the current # of iterations */
  int32_t maxniters; /* the maximum number of BCLS iterations */

  /* parameters for FSLIM neighbor finding */
  int32_t *marker;   /* used to mark the columns already seen */
  gk_fkv_t *cand;    /* keeps track of candidates and their partial scores */

} wspace_t;

#endif
