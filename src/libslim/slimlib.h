/*!
\file
\brief The top-level header file for the library

\date   Started 03/07/15
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/


#ifndef __LIBSLIM_H__
#define __LIBSLIM_H__

#include <GKlib.h>
#include <slim.h>

#if defined(ENABLE_OPENMP)
  #include <omp.h>
#endif

#include "def.h"  
#include "struct.h"
#include "macros.h"
#include "proto.h"

#if defined(COMPILER_MSC)
#if defined(rint)
  #undef rint
#endif
#define rint(x) ((idx_t)((x)+0.5))  /* MSC does not have rint() function */
#endif

#endif
