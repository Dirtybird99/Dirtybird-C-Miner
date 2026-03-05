#pragma once

// DERO Miner - Only AstroBWTv3 support
#include <astrobwtv3/astrobwtv3.h>
#include <astrobwtv3/astrotest.hpp>

#define DEFINE_UNSUPPORTED_MSG(name, algo) \
    const char* unsupported_##name = "This Binary was compiled without " algo " support... \n" \
    "Please source a DIRTYBIRD Miner binary with " algo " support";

// DERO-only algorithm support message
DEFINE_UNSUPPORTED_MSG(astro, "AstroBWTv3")