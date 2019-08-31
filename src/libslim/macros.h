/*!
\file
\brief Various macro definitions

\date   Started 3/11/2015
\author George & Xia
\author Copyright 2019, Regents of the University of Minnesota
*/

#ifndef _LIBSLIM_MACROS_H_
#define _LIBSLIM_MACROS_H_

/* gets the appropriate option value */
#define GETOPTION(options, idx, defval) \
            ((options) == NULL || (options)[idx] == -1 ? defval : (options)[idx]) 


#endif
