/*!
\file
\brief The top-level header file for the library

\date   Started 03/11/15
\author George & Xia
\author Copyright 2019, Regents of the University of Minnesota
*/


#ifndef __SLIMBIN_H__
#define __SLIMBIN_H__

#include <GKlib.h>
#include <slim.h>

#if defined(ENABLE_OPENMP)
  #include <omp.h>
#endif

#include "def.h"  
#include "struct.h"
#include "proto.h"

#if defined(COMPILER_MSC)
#if defined(rint)
  #undef rint
#endif
#define rint(x) ((idx_t)((x)+0.5))  /* MSC does not have rint() function */
#endif

#endif
