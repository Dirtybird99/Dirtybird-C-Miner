#pragma once

#include <cstddef>
#include <cstdint>

/* True when the CPU has SHA-NI (plus SSSE3 and SSE4.1); cached after the first
 * call. */
bool dluna_sha256_x2_available();

/* Hash two independent messages with interleaved SHA-NI streams.
 *
 * There is no software fallback: the body executes SHA-NI unconditionally, so a
 * caller that skips dluna_sha256_x2_available() takes a #UD on a CPU without the
 * extension. The two lengths are independent; the shorter message idles through
 * the trailing blocks of the longer one. */
void dluna_sha256_x2(const uint8_t* in_a, size_t len_a, uint8_t out_a[32],
                     const uint8_t* in_b, size_t len_b, uint8_t out_b[32]);
