#include "miners.hpp"

#define blankfunc(S) void S(int tid){}

// AstroBWTv3 coins
#ifndef DIRTYBIRD_ASTROBWTV3
blankfunc(mineDero);
blankfunc(mineSpectre);
#endif

// XelisHash v1/v2 coins
#if !defined(DIRTYBIRD_XELISHASH) || defined(DIRTYBIRD_HIP)
blankfunc(mineXelis);
#endif

// RandomX coins
#ifndef DIRTYBIRD_RANDOMX
blankfunc(mineRx0);
#endif

// Verus 
#ifndef DIRTYBIRD_VERUSHASH
blankfunc(mineVerus);
#endif

// Astrix
#ifndef DIRTYBIRD_ASTRIXHASH
blankfunc(mineAstrix);
#endif

// Nexellia
#ifndef DIRTYBIRD_NXLHASH
blankfunc(mineNexellia);
#endif

// Hoosat
#ifndef DIRTYBIRD_HOOHASH
blankfunc(mineHoosat);
#endif

// Hoosat
#ifndef DIRTYBIRD_WALAHASH
blankfunc(mineWaglayla);
#endif

// Shai
#ifndef DIRTYBIRD_SHAIHIVE
blankfunc(mineShai);
#endif

// Yespower
#ifndef DIRTYBIRD_YESPOWER
blankfunc(mineYespower);
#endif

// Rinhash
#ifndef DIRTYBIRD_RINHASH
blankfunc(mineRinhash);
#endif

#ifndef DIRTYBIRD_HIP
blankfunc(mineAstrix_hip);
blankfunc(mineNexellia_hip);
blankfunc(mineWaglayla_hip);
#endif
