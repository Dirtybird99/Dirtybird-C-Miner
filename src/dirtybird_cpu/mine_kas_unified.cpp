#include "../coins/miners.hpp"

#ifndef DIRTYBIRD_HIP
#include "common/mine_cpu_unified.hpp"

#ifdef DIRTYBIRD_ASTRIXHASH
#include "algos/astrix_cpu.hpp"
#endif

#ifdef DIRTYBIRD_NXLHASH
#include "algos/nexellia_cpu.hpp"
#endif

#ifdef DIRTYBIRD_WALAHASH
#include "algos/waglayla_cpu.hpp"
#endif

#ifdef DIRTYBIRD_HOOHASH
#include "algos/hoosat_cpu.hpp"
#endif

#ifdef DIRTYBIRD_ASTRIXHASH
// Wrapper function for Astrix unified mining
void mineAstrix_unified(int tid) {
    mineCPU_unified(tid, "astrix");
}
#endif

#ifdef DIRTYBIRD_NXLHASH
// Wrapper function for Nexellia unified mining
void mineNexellia_unified(int tid) {
    mineCPU_unified(tid, "nexellia");
}
#endif

#ifdef DIRTYBIRD_WALAHASH
// Wrapper function for Waglayla unified mining
void mineWaglayla_unified(int tid) {
    mineCPU_unified(tid, "waglayla");
}
#endif

#ifdef DIRTYBIRD_HOOHASH
// Wrapper function for Hoosat unified mining
void mineHoosat_unified(int tid) {
    mineCPU_unified(tid, "hoosat");
}
#endif

#endif // !DIRTYBIRD_HIP
