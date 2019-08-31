/*!
\file
\brief Parsing of command-line arguments

\date   Started 3/10/2015
\author George & Xia
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
                                          {"optTol", 1, 0, CMD_OPTTOL},
                                          {"niters", 1, 0, CMD_NITERS},
                                          {"nnbrs", 1, 0, CMD_NNBRS},
                                          {"simtype", 1, 0, CMD_SIMTYPE},
                                          {"nrcmds", 1, 0, CMD_NRCMDS},
                                          {"ordered", 0, 0, CMD_ORDERED},
                                          {"nthreads", 1, 0, CMD_NTHREADS},
                                          {"dbglvl", 1, 0, CMD_DBGLVL},
                                          {"help", 0, 0, CMD_HELP},
                                          {"algo", 1, 0, CMD_ALGO},
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
    "   slim_mselect [options] train-file test-file l12-file",
    " ",
    " Parameters:",
    "   train-file",
    "       The file that stores the training data.",
    " ",
    "   test-file",
    "       The file that stores the testing data.",
    " ",
    "   l12-file",
    "       The file that stores the sets of l1 and l2 regularization",
    "       parameters over which to search for models.",
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
    "   -nnbrs=int",
    "      Selects FSLIM model and specifies the number of item nearest "
    "neighbors",
    "      to be used. The default value is 0.",
    " ",
    "   -simtype=string",
    "      Specifies the similarity function for determining the neighbors. ",
    "      Available options are:",
    "        cos     -  cosine similarity [default].",
    "        jac     -  extended Jacquard similarit.",
    "        dotp    -  dot-product similarity.",
    " ",
    "   -algo=string",
    "      Specifies the optimization algorithms for learning the model. ",
    "      Available options are:",
    "        admm     -  ADMM.",
    "        cd       -  Coordinate Descent.",
    " ",
    "   -optTol=float",
    "      Specifies the threshold used during optimization for termination.",
    "      The default value is 1e-7.",
    " ",
    "   -niters=int",
    "      Specifies the maximum number of allowed optimization iterations. ",
    "      The default value is 10000.",
    " ",
    "   -nrcmds=int",
    "      Selects the number of items to recommend. The default value is 0.",
    " ",
    "   -nthreads=int",
    "      Specifies the number of threads to be used for estimation.",
    "      The default value is maximum number of threads available on the "
    "machine.",
    " ",
    "   -dbglvl=int",
    "      Specifies the debug level. The default value is 0.",
    " ",
    "   -help",
    "      Prints this message.",
    " ",
    ""};

/*------------------------------------------------------------------------*/
/*! A short help */
/*------------------------------------------------------------------------*/
static char shorthelpstr[][100] = {
    " ", " Usage: slim_mselect [options] train-file test-file l12-file",
    "   use 'slim_mselect -help' for a summary of the options.", ""};

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
  params->optTol = 1e-7;
  params->niters = 10000;
  params->nnbrs = 0;
  params->simtype = SLIM_SIMTYPE_COS;
  params->nrcmds = 10;
  params->algo = SLIM_ALGO_CD;
  params->nthreads = 1;
#ifdef __OPENMP__
  params->nthreads = omp_get_max_threads();
#endif
  params->ordered = 0;
  params->dbglvl = 0;

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

    case CMD_DBGLVL:
      if ((params->dbglvl = atoi(gk_optarg)) < 0)
        errexit("The -dbglvl parameter should be non-negative.\n");
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

    case CMD_NRCMDS:
      if ((params->nrcmds = atoi(gk_optarg)) < 0)
        errexit("The -nrcmds parameter should be non-negative.\n");
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
  if (argc - gk_optind != 3) {
    for (int i = 0; strlen(shorthelpstr[i]) > 0; i++)
      printf("%s\n", shorthelpstr[i]);
    exit(0);
  }

  params->trnfile = gk_strdup(argv[gk_optind++]);
  if (!gk_fexists(params->trnfile))
    errexit("Input training file %s does not exist.\n", params->trnfile);

  params->tstfile = gk_strdup(argv[gk_optind++]);
  if (!gk_fexists(params->tstfile))
    errexit("Input test file %s does not exist.\n", params->tstfile);

  params->l12file = gk_strdup(argv[gk_optind++]);
  if (!gk_fexists(params->l12file))
    errexit("Input l12 file %s does not exist.\n", params->l12file);

  return params;
}
