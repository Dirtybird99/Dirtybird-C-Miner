#include "lookup_full.hpp"

#include "lookup_tables.hpp"

#include <cstdint>

namespace lookup_full {

uint8_t* g_lookup3D = nullptr;

void generate_lookup3D(uint8_t* table) {
  if (table == nullptr) {
    return;
  }

  for (int op = 0; op < 256; ++op) {
    const uint32_t opcode = CodeLUT[op];
    const size_t op_base = static_cast<size_t>(op) << 16;

    for (int pos2 = 0; pos2 < 256; ++pos2) {
      const size_t row_base = op_base + (static_cast<size_t>(pos2) << 8);
      for (int input = 0; input < 256; ++input) {
        table[row_base + static_cast<size_t>(input)] =
            lookup_tables::computeBranchCorrect(
                static_cast<uint8_t>(input),
                static_cast<uint8_t>(pos2),
                opcode);
      }
    }
  }
}

}  // namespace lookup_full
