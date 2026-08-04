#!/usr/bin/env python3
"""Prove the 2-way SHA-NI compress kept its legacy-SSE encoding.

One VEX instruction inside the compress body would put the core in the AVX
state and make every legacy sha256rnds2 pay an AVX->SSE transition (~70
cycles/round, a ~50x kernel slowdown), so this is a release gate, not a lint.

The binary also contains the single-stream SHA-NI kernel, whose intrinsics
legitimately compile to VEX moves between rounds under -mavx2. The two-lane
kernel is therefore identified structurally: a maximal VEX-free instruction
run containing all 64 sha256rnds2 of one compress body. At least one such run
must exist.

Usage: check_sha_x2_novex.py <binary> [llvm-objdump-path]
"""
import re
import subprocess
import sys

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    binary = sys.argv[1]
    objdump = sys.argv[2] if len(sys.argv) > 2 else "llvm-objdump"

    dis = subprocess.run([objdump, "-d", binary],
                         capture_output=True, text=True).stdout
    rows = []
    for line in dis.splitlines():
        m = re.match(r"\s*[0-9a-f]+:\s+((?:[0-9a-f]{2} )+)\s*(\S+)", line)
        if m:
            rows.append((m.group(1).split()[0], m.group(2)))

    total_rnds2 = sum(1 for _, mnem in rows if mnem == "sha256rnds2")
    if total_rnds2 == 0:
        print("FAIL: no sha256rnds2 in %s (kernel compiled out?)" % binary)
        return 1

    # Split the stream at VEX instructions (C4/C5 prefix byte or v-mnemonic,
    # vzeroupper excepted -- it is the deliberate state reset).
    best = 0
    seg = 0
    segments_64 = 0
    for first_byte, mnem in rows:
        is_vex = (first_byte in ("c4", "c5") or
                  (mnem.startswith("v") and mnem != "vzeroupper"))
        if is_vex:
            if seg == 64:
                segments_64 += 1
            best = max(best, seg)
            seg = 0
        elif mnem == "sha256rnds2":
            seg += 1
    if seg == 64:
        segments_64 += 1
    best = max(best, seg)

    print("sha256rnds2 total=%d, largest VEX-free run=%d, intact 64-round "
          "segments=%d" % (total_rnds2, best, segments_64))
    if segments_64 >= 1:
        print("PASS: legacy-SSE two-lane compress intact")
        return 0
    print("FAIL: no VEX-free 64-round segment -- VEX leaked into the "
          "two-lane compress")
    return 1

if __name__ == "__main__":
    sys.exit(main())
