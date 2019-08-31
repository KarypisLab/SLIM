#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 01:36:28 2019

@author: dminerx007
"""

import numpy as np

# The maximum length of the options[] array
SLIM_NOPTIONS = 40

# return codes
SLIM_OK = 1             # Returned normally
SLIM_ERROR_INPUT = -2   # Returned due to erroneous inputs and/or options
SLIM_ERROR_MEMORY = -3  # Returned due to insufficient memory
SLIM_ERROR = -4         # Some other errors

# the type of model
SLIM_MTYPE_SLIM = 0    # SLIM model
SLIM_MTYPE_FSLIM = 1   # FSLIM model
SLIM_MTYPE_OSLIM = 2   # OSLIM model
SLIM_MTYPE_OFSLIM = 3  # OFSLIM model

# the type of similarities
SLIM_SIMTYPE_COS = 0   # cosine similarity
SLIM_SIMTYPE_JAC = 1   # extended Jackard similarity
SLIM_SIMTYPE_DOTP = 2  # dot-product similarity

# the optimization algorithms
SLIM_ALGO_ADMM = 0  # ADMM
SLIM_ALGO_CD = 1    # Coordinate Descent

# Options codes (i.e., options[])
SLIM_OPTION_DBGLVL=0     # Level of debuging output
SLIM_OPTION_NNBRS=1      # The number of pre-computed nearest neirbors for FSLIM
SLIM_OPTION_SIMTYPE=2    # The similarity type for FSLIM
SLIM_OPTION_NTHREADS=3   # The number of OpenMP threads to used for computation
SLIM_OPTION_MAXNITERS=4  # The maximum number of optimization iterations
SLIM_OPTION_ALGO=5       # The optimization algorithm used to learn the model
SLIM_OPTION_ORDERED=6    # A value of one assumes that the items are rated in the specified order and SSLIM is used
SLIM_OPTION_L1R=7        # The L1 regularization
SLIM_OPTION_L2R=8        # The L2 regularization
SLIM_OPTION_OPTTOL=9     # The tolerance for the solver
SLIM_OPTION_NRCMDS=10    # The number of items to be recommended

# debug levels
SLIM_DBG_INFO = 1        # Shows various diagnostic messages
SLIM_DBG_TIME = 2        # Perform timing analysis
SLIM_DBG_PROGRESS = 4    # Show progress information
SLIM_DBG_PROGRESS2 = 16  # Show more detailed progress information
SLIM_DBG_MEMORY = 2048   # Show info related to wspace allocation

# The type of model
slim_mtype_et = {
  'slim':0,       # /*!< SLIM model */
  'fslim':1,      # /*!< FSLIM model */
  'oslim':2,      # /*!< OSLIM model */
  'ofslim':3,     # /*!< OFSLIM model */
}

# The type of similarities
slim_simtype_et = {
  'cos':0,      # /*!< cosine similarity */
  'jac':1,      # /*!< extended Jackard similarity */
  'dotp':2      # /*!< dot-product similarity */
} 

# The optimization algorithms
slim_algo_et = {
  'admm':0,     # /*!< ADMM */
  'cd':1,       # /*!< Coordinate Descent */
}

array_1d_double_t = np.ctypeslib.ndpointer(dtype=np.double, ndim=1, flags='CONTIGUOUS')
array_1d_float32_t = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='CONTIGUOUS')
array_1d_ssize_t = np.ctypeslib.ndpointer(dtype=np.intp, ndim=1, flags='CONTIGUOUS')
array_1d_int32_t = np.ctypeslib.ndpointer(dtype=np.int32, ndim=1, flags='CONTIGUOUS')

def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func