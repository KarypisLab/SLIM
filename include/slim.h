/**
 * @file
 * @brief The user-level include file for the SLIM library
 * @date   Started 03/07/15
 * @author George & Xia
 * author Copyright 2011-2015, Regents of the University of Minnesota
 */

#ifndef __slim_h__
#define __slim_h__

/* Uniform definitions for various compilers */
#if defined(_MSC_VER)
#define COMPILER_MSC
#endif
#if defined(__ICC)
#define COMPILER_ICC
#endif
#if defined(__GNUC__)
#define COMPILER_GCC
#endif

/* Include c99 int definitions and need constants. When building the library,
 * these are already defined by GKlib; hence the test for _GKLIB_H_ */
#ifndef _GKLIB_H_
#ifdef COMPILER_MSC
#include <limits.h>

typedef __int32 int32_t;
typedef __int64 int64_t;
#define PRId32 "I32d"
#define PRId64 "I64d"
#define SCNd32 "ld"
#define SCNd64 "I64d"
#define INT32_MIN ((int32_t)_I32_MIN)
#define INT32_MAX _I32_MAX
#define INT64_MIN ((int64_t)_I64_MIN)
#define INT64_MAX _I64_MAX
#else
#include <inttypes.h>
#endif
#endif

/*------------------------------------------------------------------------
 * Setup the basic datatypes
 *-------------------------------------------------------------------------*/
typedef void slim_t;

/*------------------------------------------------------------------------
 * Constant definitions
 *-------------------------------------------------------------------------*/
/* SLIM's version number */
#define SLIM_VERSION "2.0"

/* The maximum length of the options[] array */
#define SLIM_NOPTIONS 40

/*------------------------------------------------------------------------
 * Function prototypes
 *-------------------------------------------------------------------------*/
#ifdef _WINDLL
#define SLIM_API(type) __declspec(dllexport) type __cdecl
#elif defined(__cdecl)
#define SLIM_API(type) type __cdecl
#else
#define SLIM_API(type) type
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Sets the default value (-1) for passed options
 *
 * @param options the integer array having option values
 * @return 1 on success
 */
SLIM_API(int32_t)
SLIM_iSetDefaults(int32_t *options);

/**
 * @brief Sets the default value (-1) for passed options
 *
 * @param options the double array having option values
 * @return 1 on success
 */
SLIM_API(int32_t)
SLIM_dSetDefaults(double *options);

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
SLIM_API(slim_t *)
SLIM_Learn(int32_t nrows, ssize_t *rowptr, int32_t *rowind, float *rowval,
           int32_t *ioptions, double *doptions, slim_t *imodel,
           int32_t *r_status);

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
SLIM_API(int32_t)
SLIM_GetTopN(slim_t *model, int32_t nratings, int32_t *itemids, float *ratings,
             int32_t *ioptions, int32_t nrcmds, int32_t *rids, float *rscores);

/**
 * @brief Writes the model to a supplied file.
 *
 * @param model the SLIM model matrix
 * @param filename the name of the file to write model to in CSR format
 * @return 1 on success
 */
SLIM_API(int32_t)
SLIM_WriteModel(slim_t *model, char *filename);

/**
 * @brief Reads the model from the passed file in CSR format
 *
 * @param filename the name of the file having model in CSR format
 * @return return the SLIM model sparse matrix in CSR format
 */
SLIM_API(slim_t *)
SLIM_ReadModel(char *filename);

/**
 * @brief frees the memory allocated by the SLIM model matrix
 *
 * @param model the SLIM model sparse matrix
 */
SLIM_API(void)
SLIM_FreeModel(slim_t **model);

/* for internal use  */

/**
 * @brief  Returns an array marking as 0 the columns that belong to the head and
 * as 1 the columns that belong to the tail. The split is based on an 50-50
 * split (head: the most frequent items that correspond to the 50% of the
 * ratings).
 *
 */
SLIM_API(int32_t *)
SLIM_DetermineHeadAndTail(int32_t nrows, int32_t ncols, ssize_t *rowptr,
                          int32_t *rowind);

#ifdef __cplusplus
}
#endif

/*------------------------------------------------------------------------
 * Enum type definitions
 *-------------------------------------------------------------------------*/
/*! Return codes */
typedef enum {
  SLIM_OK = +1,          /*!< Returned normally */
  SLIM_ERROR_INPUT = -2, /*!< Returned due to erroneous inputs and/or options */
  SLIM_ERROR_MEMORY = -3, /*!< Returned due to insufficient memory */
  SLIM_ERROR = -4,        /*!< Some other errors */
} slim_rstatus_et;

/*! The type of model */
typedef enum {
  SLIM_MTYPE_SLIM = 0,   /*!< SLIM model */
  SLIM_MTYPE_FSLIM = 1,  /*!< FSLIM model */
  SLIM_MTYPE_OSLIM = 2,  /*!< OSLIM model */
  SLIM_MTYPE_OFSLIM = 3, /*!< OFSLIM model */
} slim_mtype_et;

/* The text labels for the different simtypes */
static char slim_mtypenames[][10] = {"SLIM", "FSLIM", "OSLIM", "OFSLIM", ""};

/*! The type of similarities */
typedef enum {
  SLIM_SIMTYPE_COS = 0, /*!< cosine similarity */
  SLIM_SIMTYPE_JAC = 1, /*!< extended Jackard similarity */
  SLIM_SIMTYPE_DOTP = 2 /*!< dot-product similarity */
} slim_simtype_et;

/* The text labels for the different simtypes */
static char slim_simtypenames[][10] = {"cos", "jac", "dotp", ""};

/*! The optimization algorithms */
typedef enum {
  SLIM_ALGO_ADMM = 0, /*!< ADMM */
  SLIM_ALGO_CD = 1,   /*!< Coordinate Descent */
} slim_algo_et;

/* The text labels for the different simtypes */
static char slim_algonames[][10] = {"admm", "cd", ""};

/*! Options codes (i.e., options[]) */
typedef enum {
  SLIM_OPTION_DBGLVL, /*!< Level of debuging output */
  SLIM_OPTION_NNBRS,  /*!< The number of pre-computed nearest neirbors for FSLIM
                       */
  SLIM_OPTION_SIMTYPE,   /*!< The similarity type for FSLIM */
  SLIM_OPTION_NTHREADS,  /*!< The number of OpenMP threads to used for
                            computation */
  SLIM_OPTION_MAXNITERS, /*!< The maximum number of optimization iterations */
  SLIM_OPTION_ALGO, /*!< The optimization algorithm used to learn the model */
  SLIM_OPTION_ORDERED, /*!< A value of one assumes that the items are rated in
                            the specified order and SSLIM is used */
  SLIM_OPTION_L1R,     /*!< The L1 regularization */
  SLIM_OPTION_L2R,     /*!< The L2 regularization */
  SLIM_OPTION_OPTTOL,  /*!< The tolerance for the solver */
  SLIM_OPTION_NRCMDS   /*!< The number of items to be recommended */
} slim_options_et;

/*! Debug Levels */
typedef enum {
  SLIM_DBG_INFO = 1,       /*!< Shows various diagnostic messages */
  SLIM_DBG_TIME = 2,       /*!< Perform timing analysis */
  SLIM_DBG_PROGRESS = 4,   /*!< Show progress information */
  SLIM_DBG_PROGRESS2 = 16, /*!< Show more detailed progress information */
  SLIM_DBG_MEMORY = 2048,  /*!< Show info related to wspace allocation */
} slim_dbglvl_et;

#endif
