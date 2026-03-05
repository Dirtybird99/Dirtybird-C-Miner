#pragma once

#include <cstdint>

struct Sha256OverrideTelemetrySnapshot {
  uint64_t init_calls;
  uint64_t update_calls;
  uint64_t final_calls;
  uint64_t update_bytes;
  uint64_t update_ns;
  uint64_t final_ns;
  uint64_t final_zeroize_ns;
  uint64_t update_len_le32_calls;
  uint64_t update_len_33_48_calls;
  uint64_t update_len_49_64_calls;
  uint64_t update_len_65_128_calls;
  uint64_t update_len_129_512_calls;
  uint64_t update_len_gt512_calls;
  uint64_t update_flush_blocks;
  uint64_t update_direct_blocks;
  uint64_t final_blocks;
};

void resetSha256OverrideTelemetry();
Sha256OverrideTelemetrySnapshot getSha256OverrideTelemetrySnapshot();
