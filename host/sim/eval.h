// eval.h - simulator scenarios that measure the whole system in closed loop.
#pragma once

#include "scenario.h"

int evalTune(Domain domain = Domain::NOMINAL);  // per-surface damping characterisation -> curve targets

// damping strategies: comfort, protection, servo wear, reaction, detection speed.
// classify_ms <= 0 uses the App's default classification period.
int evalRoad(int runs, Domain domain = Domain::NOMINAL, int classify_ms = 0);
int evalObstacle();               // step vs Type-1 vs interval Type-2 fuzzy near obstacles
int evalFaults();                 // sensor failures while driving
int evalEkf(int runs);            // classifier-adaptive EKF vs fixed-noise EKF vs simple estimates
int evalMaintenance();            // shock wear over 40 simulated days
int evalUncertainty(int runs);    // behaviour on a surface never seen in training
int runLive(double speed);        // real-time loop for the dashboard (stdin commands, stdout telemetry)
int recordSurface(const char* surface, double seconds, unsigned seed);  // raw log like 'log raw'
