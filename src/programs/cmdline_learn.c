/*!
\file
\brief Parsing of command-line arguments

\date   Started 3/10/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimbin.h"

#ifdef __OPENMP__
#include <omp.h>
#endif

/*------------------------------------------------------------------------*/
/*! Command-line options */
/*------------------------------------------------------------------------*/
static struct gk_option long_options[] = {{"ifmt", 1, 0, CMD_IFMT},
                                          {"binarize", 0, 0, CMD_BINARIZE},
                                          {"l1r", 1, 0, CMD_L1R},
                                          {"l2r", 1, 0, CMD_L2R},
                                          {"optTol", 1, 0, CMD_OPTTOL},
                                          {"niters", 1, 0, CMD_NITERS},
                                          {"nnbrs", 1, 0, CMD_NNBRS},
                                          {"simtype", 1, 0, CMD_SIMTYPE},
                                          {"algo", 1, 0, CMD_ALGO},
                                          {"ordered", 0, 0, CMD_ORDERED},
                                          {"nthreads", 1, 0, CMD_NTHREADS},
                                          {"ipmdlfile", 1, 0, CMD_IPMDLFILE},
                                          {"dbglvl", 1, 0, CMD_DBGLVL},
                                          {"help", 0, 0, CMD_HELP},
                                          {0, 0, 0, 0}};

/*------------------------------------------------------------------------*/
/*! Mappings for the various parameter values */
/*------------------------------------------------------------------------*/
static gk_StringMap_t ifmt_options[] = {
    {"csr", GK_CSR_FMT_CSR},
    {"csrnv", GK_CSR_FMT_METIS}, /* this will be converted to CSR */
    {"cluto", GK_CSR_FMT_CLUTO},
    {"ijv", GK_CSR_FMT_IJV},
    {NULL, 0}};

static gk_StringMap_t simtype_options[] = {{"cos", SLIM_SIMTYPE_COS},
                                           {"jac", SLIM_SIMTYPE_JAC},
                                           {"dotp", SLIM_SIMTYPE_DOTP},
                                           {NULL, 0}};

static gk_StringMap_t algo_options[] = {
    {"admm", SLIM_ALGO_ADMM}, {"cd", SLIM_ALGO_CD}, {NULL, 0}};

/*------------------------------------------------------------------------*/
/*! Mini help */
/*------------------------------------------------------------------------*/
static char helpstr[][512] = {
    " ",
    " Usage:",
    "   slim_learn [options] train-file [model-file]",
    " ",
    " Parameters:",
    "   train-file",
    "       The file that stores the training data.",
    " ",
    "   model-file",
    "       The file that will store the model. If no model-file name is provided,",
    "       a model named slim.model will be saved.",
    " ",
    " Options:",
    "   -ifmt=string",
    "      Specifies the format of the input file. Available options are:",
    "        csr     -  CSR format [default].",
    "        csrnv   -  CSR format without ratings.",
    "        cluto   -  Format used by CLUTO.",
    "        ijv     -  One (row#, col#, val) per line.",
    " "
    "   -binarize",
    "      Specifies that the ratings should be binarized.",
    " ",
    "   -l1r=double",
    "      Specifies the L1 regularization parameter. The default value is "
    "1.0.",
    " ",
    "   -ipmdlfile=string",
    "      Specifies the file used to initialize the model.",
    " ",
    "   -l2r=double",
    "      Specifies the L2 regularization parameter. The default value is "
    "1.0.",
    " ",
    "   -nnbrs=int",
    "      Selects FSLIM model and specifies the number of item nearest "
    "neighbors",
    "      to be used. The default value is 0.",
    " ",
    "   -simtype=string",
    "      Specifies the similarity function for determining the neighbors. ",
    "      Available options are:",
    "        cos     -  cosine similarity [default].",
    "        jac     -  extended Jaccard similarity.",
    "        dotp    -  dot-product similarity.",
    " ",
    "   -algo=string",
    "      Specifies the optimization algorithms for learning the model. ",
    "      Available options are:",
    "        admm     -  ADMM.",
    "        cd       -  Coordinate Descent [default].",
    " ",
    "   -optTol=float",
    "      Specifies the threshold used during optimization for termination.",
    "      The default value is 1e-7.",
    " ",
    "   -niters=int",
    "      Specifies the maximum number of allowed optimization iterations. ",
    "      The default value is 10000.",
    " ",
    "   -nthreads=int",
    "      Specifies the number of threads to be used for estimation.",
    "      The default value is maximum number of threads available in the "
    "machine.",
    " ",
    "   -dbglvl=int",
    "      Specifies the debug level. The default value turns on info and "
    "timing.",
    " ",
    "   -help",
    "      Prints this message.",
    " ",
    ""};

/*------------------------------------------------------------------------*/
/*! A short help */
/*------------------------------------------------------------------------*/
static char shorthelpstr[][100] = {
    " ", " Usage: slim_learn [options] train-file [model-file]",
    "   use 'slim_learn -help' for a summary of the options.", ""};

/**************************************************************************/
/*! Parses command-line arguments */
/**************************************************************************/
params_t *parse_cmdline(int argc, char *argv[]) {
  int c = -1, option_index = -1;
  params_t *params;

  params = (params_t *)gk_malloc(sizeof(params_t), "parse_cmdline");
  memset((void *)params, 0, sizeof(params_t));

  /* setup defaults */
  params->ifmt = GK_CSR_FMT_CSR;
  params->readvals = 1;
  params->binarize = 0;
  params->mdlfile = NULL;
  params->ipmdlfile = NULL;
  params->l1r = 1.0;
  params->l2r = 1.0;
  params->optTol = 1e-7;
  params->niters = 10000;
  params->nnbrs = 0;
  params->simtype = SLIM_SIMTYPE_COS;
  params->algo = SLIM_ALGO_CD;
  params->nthreads = 1;
#ifdef __OPENMP__
  params->nthreads = omp_get_max_threads();
#endif
  params->ordered = 0;
  params->dbglvl = SLIM_DBG_INFO | SLIM_DBG_TIME;

  while ((c = gk_getopt_long_only(argc, argv, "", long_options,
                                  &option_index)) != -1) {
    switch (c) {
    case CMD_IFMT:
      if ((params->ifmt = gk_GetStringID(ifmt_options, gk_optarg)) == -1)
        errexit("Invalid -ifmt of %s.\n", gk_optarg);
      /* deal with the no-ratings case */
      if (params->ifmt == GK_CSR_FMT_METIS) {
        params->ifmt = GK_CSR_FMT_CSR;
        params->readvals = 0;
      }
      break;

    case CMD_BINARIZE:
      params->binarize = 1;
      break;

    case CMD_OUTFILE:
      params->outfile = gk_strdup(gk_optarg);
      break;

    case CMD_IPMDLFILE:
      params->ipmdlfile = gk_strdup(gk_optarg);
      break;

    case CMD_DBGLVL:
      if ((params->dbglvl = atoi(gk_optarg)) < 0)
        errexit("The -dbglvl parameter should be non-negative.\n");
      break;

    case CMD_L1R:
      if ((params->l1r = atof(gk_optarg)) < 0.0)
        errexit("The -l1r parameter should be non-negative.\n");
      break;

    case CMD_L2R:
      if ((params->l2r = atof(gk_optarg)) < 0.0)
        errexit("The -l1r parameter should be non-negative.\n");
      break;

    case CMD_OPTTOL:
      if ((params->optTol = atof(gk_optarg)) < 0.0)
        errexit("The -optTol parameter should be non-negative.\n");
      break;

    case CMD_NITERS:
      if ((params->niters = atoi(gk_optarg)) < 0)
        errexit("The -niters parameter should be non-negative.\n");
      break;

    case CMD_NTHREADS:
      if ((params->nthreads = atoi(gk_optarg)) < 0)
        errexit("The -nthreads parameter should be non-negative.\n");
      break;

    case CMD_NNBRS:
      if ((params->nnbrs = atoi(gk_optarg)) < 0)
        errexit("The -nnbrs parameter should be non-negative.\n");
      break;

    case CMD_SIMTYPE:
      if ((params->simtype = gk_GetStringID(simtype_options, gk_optarg)) == -1)
        errexit("Invalid -simtype of %s.\n", gk_optarg);
      break;

    case CMD_ALGO:
      if ((params->algo = gk_GetStringID(algo_options, gk_optarg)) == -1)
        errexit("Invalid -algo of %s.\n", gk_optarg);
      break;

    case CMD_ORDERED:
      params->ordered = 1;
      break;

    case '?':
    case CMD_HELP:
      for (int i = 0; strlen(helpstr[i]) > 0; i++)
        printf("%s\n", helpstr[i]);
      exit(0);

    default:
      printf("Illegal command-line option(s) %s\n", gk_optarg);
      exit(0);
    }
  }

  /* get the datafile */
  if (argc - gk_optind < 1 || argc - gk_optind > 2) {
    for (int i = 0; strlen(shorthelpstr[i]) > 0; i++)
      printf("%s\n", shorthelpstr[i]);
    exit(0);
  }

  params->trnfile = gk_strdup(argv[gk_optind++]);
  if (!gk_fexists(params->trnfile))
    errexit("Input training file %s does not exist.\n", params->trnfile);

  if (argc - gk_optind == 1) {
    params->mdlfile = gk_strdup(argv[gk_optind++]);
  } else {
    params->mdlfile = "slim.model";
  }

  return params;
}
