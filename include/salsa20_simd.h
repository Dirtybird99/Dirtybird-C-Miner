/**
 * SIMD-optimized Salsa20 implementation
 * Ported from TNN-miner (MIT licensed)
 *
 * Provides runtime dispatch to SSE2/AVX2/AVX512 implementations
 * based on CPU capabilities.
 */

#ifndef SALSA20_SIMD_H
#define SALSA20_SIMD_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SALSA20_STATE_SIZE 64
#define SALSA20_BLOCK_SIZE 64
#define SALSA20_KEY_SIZE 32
#define SALSA20_IV_SIZE 8

/**
 * Initialize CPU feature detection (call once at startup)
 */
void salsa20_simd_init(void);

/**
 * Check if SIMD Salsa20 is available
 */
bool salsa20_simd_available(void);

/**
 * Get the name of the active SIMD implementation
 */
const char* salsa20_simd_impl_name(void);

/**
 * Transform a single 64-byte Salsa20 state block
 * @param state 64-byte state buffer (modified in-place)
 * @param rounds Number of rounds (typically 20)
 */
void salsa20_simd_transform(uint8_t* state, int rounds);

/**
 * Transform 4 state blocks in parallel (for batch processing)
 * @param state0-3 Four 64-byte state buffers
 * @param rounds Number of rounds
 */
void salsa20_simd_transform4(uint8_t* state0, uint8_t* state1,
                              uint8_t* state2, uint8_t* state3, int rounds);

/**
 * Process bytes using Salsa20 stream cipher (compatible with ucstk::Salsa20)
 * @param key 32-byte key
 * @param iv 8-byte IV (can be NULL for zero IV)
 * @param input Input bytes (can be same as output for in-place)
 * @param output Output bytes
 * @param len Number of bytes to process
 */
void salsa20_simd_process(const uint8_t* key, const uint8_t* iv,
                          const uint8_t* input, uint8_t* output, size_t len);

/**
 * Generate keystream (XOR with zero input)
 * @param key 32-byte key
 * @param iv 8-byte IV
 * @param output Output buffer
 * @param len Number of bytes to generate
 */
void salsa20_simd_keystream(const uint8_t* key, const uint8_t* iv,
                            uint8_t* output, size_t len);

#ifdef __cplusplus
}

// C++ includes for wrapper class
#include <cstring>  // for memset, memcpy

// C++ wrapper class compatible with ucstk::Salsa20 interface
namespace simd {

class Salsa20 {
public:
    enum : size_t {
        VECTOR_SIZE = 16,
        BLOCK_SIZE = 64,
        KEY_SIZE = 32,
        IV_SIZE = 8
    };

    Salsa20(const uint8_t* key = nullptr) {
        memset(vector_, 0, sizeof(vector_));
        memset(iv_, 0, sizeof(iv_));
        counter_ = 0;
        if (key) setKey(key);
    }

    void setKey(const uint8_t* key) {
        if (key) memcpy(key_, key, KEY_SIZE);
    }

    void setIv(const uint8_t* iv) {
        if (iv) {
            memcpy(iv_, iv, IV_SIZE);
        } else {
            memset(iv_, 0, IV_SIZE);
        }
        counter_ = 0;
    }

    void processBytes(const uint8_t* input, uint8_t* output, size_t numBytes) {
        salsa20_simd_process(key_, iv_, input, output, numBytes);
    }

    void generateKeyStream(uint8_t output[BLOCK_SIZE]) {
        salsa20_simd_keystream(key_, iv_, output, BLOCK_SIZE);
    }

private:
    uint32_t vector_[16];
    uint8_t key_[KEY_SIZE];
    uint8_t iv_[IV_SIZE];
    uint64_t counter_;
};

} // namespace simd

#endif // __cplusplus

#endif // SALSA20_SIMD_H
