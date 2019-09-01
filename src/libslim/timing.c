/*!
\file
\brief Timing-related functions

\date Started 3/9/2015
\author George Karypis with contributions by Xia Ning, Athanasios N. Nikolakopoulos, Zeren Shui and Mohit Sharma.
\author Copyright 2019, Regents of the University of Minnesota
*/

#include "slimlib.h"

/*************************************************************************
 * This function clears the timers
 **************************************************************************/
void InitTimers(params_t *params) {
  gk_clearwctimer(params->TotalTmr);
  gk_clearwctimer(params->SetupTmr);
  gk_clearwctimer(params->LearnTmr);
  gk_clearwctimer(params->Aux1Tmr);
  gk_clearwctimer(params->Aux2Tmr);
  gk_clearwctimer(params->Aux3Tmr);
}

/*************************************************************************
 * This function prints the various timers
 **************************************************************************/
void PrintTimers(params_t *params) {
  printf(
      "\nTiming Information -------------------------------------------------");
  printf("\n Total: \t %7.3lf", gk_getwctimer(params->TotalTmr));
  if (gk_getwctimer(params->SetupTmr) > 0)
    printf("\n   Setup: \t\t %7.3lf", gk_getwctimer(params->SetupTmr));
  if (gk_getwctimer(params->LearnTmr) > 0)
    printf("\n   Learn: \t\t %7.3lf", gk_getwctimer(params->LearnTmr));

  if (gk_getwctimer(params->Aux1Tmr) > 0)
    printf("\n   Aux1Tmr: \t\t %7.3lf", gk_getwctimer(params->Aux1Tmr));
  if (gk_getwctimer(params->Aux2Tmr) > 0)
    printf("\n   Aux2Tmr: \t\t %7.3lf", gk_getwctimer(params->Aux2Tmr));
  if (gk_getwctimer(params->Aux3Tmr) > 0)
    printf("\n   Aux3Tmr: \t\t %7.3lf", gk_getwctimer(params->Aux3Tmr));

  printf("\n*******************************************************************"
         "*\n");
}
