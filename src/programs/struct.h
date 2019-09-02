/*!
\file
\brief Various data structures

\date   Started 03/11/15
\author George Karypis with contributions by Xia Ning, Athanasios N.
Nikolakopoulos, Zeren Shui and Mohit Sharma. \author Copyright 2019, Regents of
the University of Minnesota
*/

#ifndef __SLIMBINSTRUCT_H__
#define __SLIMBINSTRUCT_H__

/**************************************************************/
/*! The command-line parameters */
/**************************************************************/
typedef struct {
  char *trnfile;   /*!< the file of historical preferences */
  char *tstfile;   /*!< the file to validate the recommendations */
  char *l12file;   /*!< the file that contains the regularization values over
                      which to search */
  char *mdlfile;   /*!< the model file during prediction */
  char *outfile;   /*!< the model/predictions file */
  char *ipmdlfile; /*!< the model file used to initialize model*/

  double l1r;    /*!< the regularization parameter for L-1 norm */
  double l2r;    /*!< the regularization parameter for L-2 norm */
  int nnbrs;     /*!< the # of neighbors in the FSLIM model */
  int simtype;   /*!< the function for computing the neighbor similarity */
  int nthreads;  /*!< the number of threads to use */
  double optTol; /*!< optimality tolerance */
  int niters;  /*!< max number of iterations allowed for optimization solvers */
  int ordered; /*!< if the order of the items is significant */
  int algo;    /*!< algorithm used for training SLIM */
  int ifmt;    /*!< the input format */
  int readvals; /*!< indicates if ratings are provided */
  int binarize; /*!< indicates if the ratings data will be converted to implicit
                   feedback */
  int nrcmds;   /*!< the # of items to recommend */
  int dbglvl;   /*!< the debug level */

} params_t;

#endif
