/**
 * cloned_astrobwt.cpp - Absolute bit-identical clone of standard AstroBWTv3
 * for parity testing.
 */

#include "astrobwtv3.h"
#include "astroworker.h"
#include "lookupcompute.h"
#include "rc4_avx512.hpp"
#include "memory_optimized.hpp"
#include "fnv1a.h"
#include <xxhash64.h>
#include <highwayhash/sip_hash.h>
#include <immintrin.h>
#include <openssl/sha.h>
#include <cstring>
#include <algorithm>

extern "C" {
  #include "divsufsort.h"
  #ifdef USE_LIBSAIS
    #include "libsais.h"
  #endif
}

#ifdef USE_DLUNA_RADIX_SA
  #include "dluna_radix_sa.h"
  #define CLONED_SA_FUNCTION dluna_radix_sa::radix_sort_sa
#elif defined(USE_LIBSAIS)
  #define CLONED_SA_FUNCTION(T, SA, n, bA, bB) libsais(T, SA, n, 0, nullptr)
#else
  #define CLONED_SA_FUNCTION divsufsort
#endif

extern uint8_t *lookup1D_global;
extern uint8_t g_is_branched[256];
extern uint8_t g_reg_idx[256];
extern uint32_t CodeLUT[257];

// Standard hashSHA256 from astrobwtv3.cpp
static void hashSHA256_cl(SHA256_CTX &sha256, const byte *input, byte *digest, unsigned long inputSize)
{
  SHA256_Init(&sha256);
  SHA256_Update(&sha256, input, inputSize);
  SHA256_Final(digest, &sha256);
}

// Standard wolfCompute logic
void wolfCompute_cl(workerData &worker, int wIndex)
{
  worker.templateIdx = 0;
  uint8_t chunkCount = 1;
  int firstChunk = 0;
  uint8_t lp1 = 0, lp2 = 255;

  worker.tries[wIndex] = 0;
  for (int it = 0; it < 278; ++it)
  {
      worker.tries[wIndex]++;
      worker.random_switcher = worker.prev_lhash ^ worker.lhash ^ worker.tries[wIndex];

      worker.op = static_cast<byte>(worker.random_switcher);
      byte p1 = static_cast<byte>(worker.random_switcher >> 8);
      byte p2 = static_cast<byte>(worker.random_switcher >> 16);

      if (p1 > p2) std::swap(p1, p2);
      if (p2 - p1 > 32) p2 = p1 + ((p2 - p1) & 0x1f);

      if (worker.tries[wIndex] > 1) {
        lp1 = std::min(lp1, p1);
        lp2 = std::max(lp2, p2);
      } else {
        lp1 = p1; lp2 = p2;
      }

      worker.pos1 = p1;
      worker.pos2 = p2;

      worker.chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 1) * 256];

      if (worker.tries[wIndex] == 1) {
        worker.prev_chunk = worker.chunk;
      } else {
        worker.prev_chunk = &worker.sData[wIndex * ASTRO_SCRATCH_SIZE + (worker.tries[wIndex] - 2) * 256];
        memcpy(worker.chunk, worker.prev_chunk, 256);
      }

      if (worker.op >= 254) {
#if USE_CRYPTOGAMS_RC4_DUAL
        worker.cryptogams_rc4[wIndex].set_key(worker.prev_chunk, 256);
        RC4_set_key(&worker.key[wIndex], 256, worker.prev_chunk);
#else
        RC4_set_key(&worker.key[wIndex], 256,  worker.prev_chunk);
#endif
      }

      wolfPermute(worker.prev_chunk, worker.chunk, worker.op, worker.pos1, worker.pos2, worker);

      if (!worker.op) {
        if ((worker.pos2-worker.pos1)%2 == 1) {
          uint8_t t1 = worker.chunk[worker.pos1];
          uint8_t t2 = worker.chunk[worker.pos2];
          worker.chunk[worker.pos1] = reverse8(t2);
          worker.chunk[worker.pos2] = reverse8(t1);
        }
      }

      worker.A = (worker.chunk[worker.pos1] - worker.chunk[worker.pos2]);

      if (worker.A < 0x10) {
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = XXHash64::hash(worker.chunk, worker.pos2, 0);
      }
      if (worker.A < 0x20) {
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          worker.lhash = hash_64_fnv1a(worker.chunk, worker.pos2);
      }
      if (worker.A < 0x30) {
          worker.prev_lhash = worker.lhash + worker.prev_lhash;
          HH_ALIGNAS(16) const highwayhash::HH_U64 key2[2] = {(uint64_t)worker.tries[wIndex], worker.prev_lhash};
          worker.lhash = highwayhash::SipHash(key2, (char*)worker.chunk, worker.pos2);
      }

      if (worker.A <= 0x40)
      {
#if USE_CRYPTOGAMS_RC4_DUAL
        worker.cryptogams_rc4[wIndex].apply_keystream_256(worker.chunk);
#else
        RC4(&worker.key[wIndex], 256, worker.chunk,  worker.chunk);
#endif
        worker.astroTemplate[worker.templateIdx] = templateMarker{
          (uint8_t)(chunkCount > 1 ? lp1 : 0), (uint8_t)(chunkCount > 1 ? lp2 : 255),
          0, 0, (uint16_t)((firstChunk << 7) | chunkCount)
        };
        worker.templateIdx += (worker.tries[wIndex] > 1);
        firstChunk = worker.tries[wIndex]-1;
        lp1 = 255; lp2 = 0; chunkCount = 1;
      } else {
        chunkCount++;
      }

      worker.chunk[255] = worker.chunk[255] ^ worker.chunk[worker.pos1] ^ worker.chunk[worker.pos2];

      if (worker.tries[wIndex] > 260 + 16 || (worker.chunk[255] >= 0xf0 && worker.tries[wIndex] > 260))
      {
        break;
      }
  }

  if (chunkCount > 0) {
    worker.astroTemplate[worker.templateIdx++] = templateMarker{
      (uint8_t)(chunkCount > 1 ? lp1 : 0), (uint8_t)(chunkCount > 1 ? lp2 : 255),
      0, 0, (uint16_t)((firstChunk << 7) | chunkCount)
    };
  }

  worker.data_len = static_cast<uint32_t>((worker.tries[wIndex] - 4) * 256 + (((static_cast<uint64_t>(worker.chunk[253]) << 8) | static_cast<uint64_t>(worker.chunk[254])) & 0x3ff));
}

void AstroBWTv3_clone(byte *input, int inputLen, byte *outputhash, workerData &worker)
{
    uint8_t scratch[384] = {0};
    hashSHA256_cl(worker.sha256, input, &scratch[320], inputLen);

#if USE_SIMD_SALSA20
    salsa20_simd_process(&scratch[320], &scratch[256], worker.salsaInput, scratch, 256);
#else
    worker.salsa20.setKey(&scratch[320]);
    worker.salsa20.setIv(&scratch[256]);
    worker.salsa20.processBytes(worker.salsaInput, scratch, 256);
#endif

#if USE_CRYPTOGAMS_RC4_DUAL
    worker.cryptogams_rc4[0].set_key(scratch, 256);
    RC4_set_key(&worker.key[0], 256, scratch);
    worker.cryptogams_rc4[0].apply_keystream_256(scratch);
#else
    RC4_set_key(&worker.key[0], 256,  scratch);
    RC4(&worker.key[0], 256, scratch,  scratch);
#endif

    worker.lhash = hash_64_fnv1a_256(scratch);
    worker.prev_lhash = worker.lhash;
    worker.tries[0] = 0;
    memcpy(worker.sData, scratch, 256);

    wolfCompute_cl(worker, 0);

    // Prefix-radix backends read a few bytes past the logical end for key formation.
    // Zero the tail so clone runs remain deterministic across repeated invocations.
    memset(worker.sData + worker.data_len, 0, 16);
    CLONED_SA_FUNCTION(worker.sData, worker.sa, worker.data_len, nullptr, nullptr);

    hashSHA256_cl(worker.sha256, reinterpret_cast<byte*>(worker.sa), outputhash, worker.data_len * 4);
}
