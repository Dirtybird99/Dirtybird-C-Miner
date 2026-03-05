#pragma once

#include <cstddef>
#include <cstdint>

namespace lookup_full {

constexpr size_t LOOKUP3D_SIZE = 256u * 256u * 256u;  // 16 MB

extern uint8_t* g_lookup3D;

void generate_lookup3D(uint8_t* table);

inline uint8_t lookup_branch(uint8_t op, uint8_t pos2_val, uint8_t input) {
  return g_lookup3D[(static_cast<size_t>(op) << 16) |
                    (static_cast<size_t>(pos2_val) << 8) |
                    input];
}

}  // namespace lookup_full
