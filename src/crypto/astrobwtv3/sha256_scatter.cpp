/**
 * Scatter-Gather SHA-256 Implementation
 *
 * Implements SHA-256 with an index function interface for on-demand byte fetching.
 * Includes both standard (full W array) and optimized (rolling 16-word W buffer)
 * implementations.
 */

#include "sha256_scatter.hpp"
#include <cstring>
#include <cstdio>

namespace sha256_scatter {

// ============================================================================
// SHA-256 Constants
// ============================================================================

// Initial hash values
const uint32_t H_INIT[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

// Round constants
alignas(16) const uint32_t K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
};

// ============================================================================
// SHA-256 Helper Functions
// ============================================================================

// Right rotate
static inline uint32_t rotr(uint32_t x, int n) {
    return (x >> n) | (x << (32 - n));
}

// SHA-256 functions
static inline uint32_t ch(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (~x & z);
}

static inline uint32_t maj(uint32_t x, uint32_t y, uint32_t z) {
    return (x & y) ^ (x & z) ^ (y & z);
}

// Big Sigma 0: used in rounds (rotation amounts 2, 13, 22)
static inline uint32_t Sigma0(uint32_t x) {
    return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22);
}

// Big Sigma 1: used in rounds (rotation amounts 6, 11, 25)
static inline uint32_t Sigma1(uint32_t x) {
    return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25);
}

// Small sigma 0: used in message schedule (rotation amounts 7, 18, shift 3)
static inline uint32_t sigma0(uint32_t x) {
    return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

// Small sigma 1: used in message schedule (rotation amounts 17, 19, shift 10)
static inline uint32_t sigma1(uint32_t x) {
    return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}

// Read a big-endian 32-bit word from bytes
static inline uint32_t read_be32(const uint8_t* p) {
    return ((uint32_t)p[0] << 24) |
           ((uint32_t)p[1] << 16) |
           ((uint32_t)p[2] << 8) |
           ((uint32_t)p[3]);
}

// Write a big-endian 32-bit word to bytes
static inline void write_be32(uint8_t* p, uint32_t val) {
    p[0] = (val >> 24) & 0xFF;
    p[1] = (val >> 16) & 0xFF;
    p[2] = (val >> 8) & 0xFF;
    p[3] = val & 0xFF;
}

// Read a big-endian 32-bit word using index function
static inline uint32_t read_be32_scatter(GetByteFn get_byte, void* ctx, size_t offset) {
    return ((uint32_t)get_byte(offset, ctx) << 24) |
           ((uint32_t)get_byte(offset + 1, ctx) << 16) |
           ((uint32_t)get_byte(offset + 2, ctx) << 8) |
           ((uint32_t)get_byte(offset + 3, ctx));
}

// ============================================================================
// Context Functions
// ============================================================================

void Sha256Context::init() {
    memcpy(state, H_INIT, sizeof(H_INIT));
    total_len = 0;
    buffer_len = 0;
    memset(buffer, 0, sizeof(buffer));
}

void Sha256Context::reset() {
    init();
}

// ============================================================================
// Contiguous Buffer Helper
// ============================================================================

uint8_t get_byte_contiguous(size_t index, void* context) {
    ContiguousContext* ctx = static_cast<ContiguousContext*>(context);
    if (index < ctx->len) {
        return ctx->data[index];
    }
    return 0;  // Return 0 for out-of-bounds (padding behavior)
}

// ============================================================================
// Process Block with Full W Array (Reference Implementation)
// ============================================================================

/**
 * Process a single 64-byte block using scatter-gather access
 * Uses the standard 64-word W array approach
 */
void process_block_full(
    uint32_t* state,
    GetByteFn get_byte,
    void* context,
    size_t block_offset
) {
    uint32_t W[64];

    // Load first 16 words from message (big-endian)
    for (int i = 0; i < 16; i++) {
        W[i] = read_be32_scatter(get_byte, context, block_offset + i * 4);
    }

    // Expand message schedule (W[16] to W[63])
    for (int i = 16; i < 64; i++) {
        W[i] = sigma1(W[i - 2]) + W[i - 7] + sigma0(W[i - 15]) + W[i - 16];
    }

    // Initialize working variables
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // 64 rounds
    for (int i = 0; i < 64; i++) {
        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Update state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// ============================================================================
// Process Block with Rolling 16-Word W Buffer (Optimized Implementation)
// ============================================================================

/**
 * Process a single 64-byte block using a rolling 16-word W buffer
 *
 * Key optimization: Instead of storing all 64 W values, we use a circular
 * buffer of 16 words. The message schedule formula is:
 *   W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
 *
 * In a 16-word circular buffer with index i % 16:
 *   W[(i) % 16] = sigma1(W[(i-2) % 16]) + W[(i-7) % 16] +
 *                 sigma0(W[(i-15) % 16]) + W[(i-16) % 16]
 *
 * Note that (i-16) % 16 == i % 16, so we're updating W[i%16] in place.
 */
void process_block_rolling(
    uint32_t* state,
    GetByteFn get_byte,
    void* context,
    size_t block_offset
) {
    uint32_t W[16];  // Rolling circular buffer

    // Load first 16 words from message (big-endian)
    for (int i = 0; i < 16; i++) {
        W[i] = read_be32_scatter(get_byte, context, block_offset + i * 4);
    }

    // Initialize working variables
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // Process rounds 0-15 (use W values directly)
    for (int i = 0; i < 16; i++) {
        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Process rounds 16-63 with rolling W buffer
    // For round i (16 <= i < 64):
    //   We need: W[i-2], W[i-7], W[i-15], W[i-16]
    //   In circular buffer: W[(i-2)%16], W[(i-7)%16], W[(i-15)%16], W[(i-16)%16]
    //   Note: (i-16) % 16 == i % 16, so we update W[i%16] with new value
    for (int i = 16; i < 64; i++) {
        // Calculate indices in circular buffer
        int idx   = i & 15;           // i % 16 (where we'll store new value)
        int idx_2 = (i - 2) & 15;     // (i-2) % 16
        int idx_7 = (i - 7) & 15;     // (i-7) % 16
        int idx_15 = (i - 15) & 15;   // (i-15) % 16
        // Note: idx_16 = (i-16) & 15 == idx (since (i-16) % 16 == i % 16)

        // Compute new W value and store in circular buffer
        // W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16]
        uint32_t new_w = sigma1(W[idx_2]) + W[idx_7] + sigma0(W[idx_15]) + W[idx];
        W[idx] = new_w;

        // Perform round
        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[i] + new_w;
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Update state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// ============================================================================
// Helper: Create padded block buffer from index function
// ============================================================================

/**
 * Fill a 64-byte block buffer with data bytes, handling padding
 *
 * @param buffer Output buffer (64 bytes)
 * @param get_byte Index function
 * @param context Context for index function
 * @param block_offset Offset of this block in data
 * @param data_len Total data length (for bounds checking)
 * @param is_last_data_block True if this block contains the last data byte
 * @param add_padding True if padding should be added to this block
 * @param add_length True if length field should be added to this block
 * @param bit_length Total message length in bits
 */
static void fill_block_with_padding(
    uint8_t* buffer,
    GetByteFn get_byte,
    void* context,
    size_t block_offset,
    size_t data_len,
    bool add_padding,
    bool add_length,
    uint64_t bit_length
) {
    // Fill buffer with data bytes (or zeros if past end of data)
    for (size_t i = 0; i < 64; i++) {
        size_t byte_idx = block_offset + i;
        if (byte_idx < data_len) {
            buffer[i] = get_byte(byte_idx, context);
        } else if (byte_idx == data_len && add_padding) {
            buffer[i] = 0x80;  // Padding bit
        } else {
            buffer[i] = 0x00;  // Zero padding
        }
    }

    // Add length in last 8 bytes if requested
    if (add_length) {
        for (int i = 0; i < 8; i++) {
            buffer[56 + i] = (bit_length >> (56 - i * 8)) & 0xFF;
        }
    }
}

/**
 * Process a padded block from buffer (for final block handling)
 */
static void process_block_from_buffer(uint32_t* state, const uint8_t* buffer) {
    uint32_t W[16];

    // Load 16 words from buffer (big-endian)
    for (int i = 0; i < 16; i++) {
        W[i] = read_be32(buffer + i * 4);
    }

    // Initialize working variables
    uint32_t a = state[0];
    uint32_t b = state[1];
    uint32_t c = state[2];
    uint32_t d = state[3];
    uint32_t e = state[4];
    uint32_t f = state[5];
    uint32_t g = state[6];
    uint32_t h = state[7];

    // Rounds 0-15
    for (int i = 0; i < 16; i++) {
        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[i] + W[i];
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Rounds 16-63 with rolling buffer
    for (int i = 16; i < 64; i++) {
        int idx   = i & 15;
        int idx_2 = (i - 2) & 15;
        int idx_7 = (i - 7) & 15;
        int idx_15 = (i - 15) & 15;

        uint32_t new_w = sigma1(W[idx_2]) + W[idx_7] + sigma0(W[idx_15]) + W[idx];
        W[idx] = new_w;

        uint32_t T1 = h + Sigma1(e) + ch(e, f, g) + K[i] + new_w;
        uint32_t T2 = Sigma0(a) + maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + T1;
        d = c;
        c = b;
        b = a;
        a = T1 + T2;
    }

    // Update state
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
}

// ============================================================================
// Main Scatter-Gather SHA-256 Functions
// ============================================================================

/**
 * SHA-256 on scattered data using the rolling W buffer optimization
 */
void sha256_scatter_rolling(
    GetByteFn get_byte,
    void* context,
    size_t data_len,
    uint8_t* output
) {
    uint32_t state[8];
    memcpy(state, H_INIT, sizeof(H_INIT));

    if (data_len == 0) {
        // Special case: empty input
        // Padding: 0x80 followed by zeros, then 64-bit length (0)
        uint8_t buffer[64];
        memset(buffer, 0, 64);
        buffer[0] = 0x80;
        // Length is already 0
        process_block_from_buffer(state, buffer);
    } else {
        uint64_t bit_length = (uint64_t)data_len * 8;
        size_t num_full_blocks = data_len / 64;
        size_t remaining_bytes = data_len % 64;

        // Process full blocks directly with rolling buffer
        for (size_t i = 0; i < num_full_blocks; i++) {
            process_block_rolling(state, get_byte, context, i * 64);
        }

        // Handle final block(s) with padding
        // Padding adds: 1 byte (0x80), zeros, 8 bytes (length)
        // If remaining_bytes + 1 + 8 <= 64, we need 1 final block
        // Otherwise, we need 2 final blocks

        uint8_t buffer[64];

        if (remaining_bytes < 56) {
            // Fits in one block: data + 0x80 + padding + length
            memset(buffer, 0, 64);

            // Copy remaining data bytes
            for (size_t i = 0; i < remaining_bytes; i++) {
                buffer[i] = get_byte(num_full_blocks * 64 + i, context);
            }

            // Add padding bit
            buffer[remaining_bytes] = 0x80;

            // Add length in big-endian at bytes 56-63
            for (int i = 0; i < 8; i++) {
                buffer[56 + i] = (bit_length >> (56 - i * 8)) & 0xFF;
            }

            process_block_from_buffer(state, buffer);
        } else {
            // Need two blocks:
            // Block 1: remaining data + 0x80 + padding
            // Block 2: zeros + length

            // First block
            memset(buffer, 0, 64);
            for (size_t i = 0; i < remaining_bytes; i++) {
                buffer[i] = get_byte(num_full_blocks * 64 + i, context);
            }
            buffer[remaining_bytes] = 0x80;
            process_block_from_buffer(state, buffer);

            // Second block (zeros + length)
            memset(buffer, 0, 64);
            for (int i = 0; i < 8; i++) {
                buffer[56 + i] = (bit_length >> (56 - i * 8)) & 0xFF;
            }
            process_block_from_buffer(state, buffer);
        }
    }

    // Write output hash in big-endian
    for (int i = 0; i < 8; i++) {
        write_be32(output + i * 4, state[i]);
    }
}

/**
 * SHA-256 on scattered data (C function pointer version)
 * Uses the rolling W buffer optimization
 */
void sha256_scatter(
    GetByteFn get_byte,
    void* context,
    size_t data_len,
    uint8_t* output
) {
    sha256_scatter_rolling(get_byte, context, data_len, output);
}

/**
 * SHA-256 on scattered data (C++ std::function version)
 */
void sha256_scatter_func(
    GetByteFunc get_byte,
    size_t data_len,
    uint8_t* output
) {
    // Wrap std::function for C interface
    struct FuncContext {
        GetByteFunc* func;
    };
    FuncContext ctx{&get_byte};

    auto wrapper = [](size_t index, void* context) -> uint8_t {
        FuncContext* ctx = static_cast<FuncContext*>(context);
        return (*ctx->func)(index);
    };

    sha256_scatter_rolling(wrapper, &ctx, data_len, output);
}

// ============================================================================
// Standard SHA-256 for Verification
// ============================================================================

void sha256_standard(
    const uint8_t* data,
    size_t data_len,
    uint8_t* output
) {
    ContiguousContext ctx{data, data_len};
    sha256_scatter_rolling(get_byte_contiguous, &ctx, data_len, output);
}

// ============================================================================
// Correctness Verification
// ============================================================================

bool hashes_equal(const uint8_t* hash1, const uint8_t* hash2) {
    return memcmp(hash1, hash2, 32) == 0;
}

void hash_to_hex(const uint8_t* hash, char* hex_str) {
    const char* hex_chars = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex_str[i * 2] = hex_chars[(hash[i] >> 4) & 0xF];
        hex_str[i * 2 + 1] = hex_chars[hash[i] & 0xF];
    }
    hex_str[64] = '\0';
}

void print_hash(const uint8_t* hash) {
    char hex[65];
    hash_to_hex(hash, hex);
    printf("%s\n", hex);
}

/**
 * Verify scatter-gather implementation produces same output as standard
 */
bool verify_scatter_correctness(
    const uint8_t* data,
    size_t data_len
) {
    uint8_t hash_standard[32];
    uint8_t hash_scatter[32];

    // Compute with standard implementation
    sha256_standard(data, data_len, hash_standard);

    // Compute with scatter-gather
    ContiguousContext ctx{data, data_len};
    sha256_scatter(get_byte_contiguous, &ctx, data_len, hash_scatter);

    return hashes_equal(hash_standard, hash_scatter);
}

/**
 * Verify rolling W buffer produces same result as full W array
 */
static bool verify_rolling_vs_full(
    const uint8_t* data,
    size_t data_len
) {
    uint32_t state_rolling[8];
    uint32_t state_full[8];

    memcpy(state_rolling, H_INIT, sizeof(H_INIT));
    memcpy(state_full, H_INIT, sizeof(H_INIT));

    ContiguousContext ctx{data, data_len};

    // Only test full blocks for direct comparison
    size_t num_blocks = data_len / 64;

    for (size_t i = 0; i < num_blocks; i++) {
        process_block_rolling(state_rolling, get_byte_contiguous, &ctx, i * 64);
        process_block_full(state_full, get_byte_contiguous, &ctx, i * 64);

        // Compare after each block
        if (memcmp(state_rolling, state_full, sizeof(state_rolling)) != 0) {
            printf("Mismatch at block %zu\n", i);
            return false;
        }
    }

    return true;
}

/**
 * Run comprehensive correctness tests
 */
int run_correctness_tests() {
    int failures = 0;
    char hex[65];

    printf("=== SHA-256 Scatter-Gather Correctness Tests ===\n\n");

    // Test 1: Empty input
    // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    {
        printf("Test 1: Empty input\n");
        uint8_t hash[32];
        sha256_standard(nullptr, 0, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        const char* expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
        if (strcmp(hex, expected) != 0) {
            printf("  FAILED! Expected: %s\n", expected);
            failures++;
        } else {
            printf("  PASSED\n");
        }
    }

    // Test 2: "abc"
    // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
    {
        printf("\nTest 2: \"abc\"\n");
        const uint8_t data[] = {'a', 'b', 'c'};
        uint8_t hash[32];
        sha256_standard(data, 3, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        const char* expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
        if (strcmp(hex, expected) != 0) {
            printf("  FAILED! Expected: %s\n", expected);
            failures++;
        } else {
            printf("  PASSED\n");
        }
    }

    // Test 3: Exactly 55 bytes (boundary: padding fits in one block)
    {
        printf("\nTest 3: 55 bytes (single padding block boundary)\n");
        uint8_t data[55];
        memset(data, 'A', 55);
        uint8_t hash[32];
        sha256_standard(data, 55, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        // Verify scatter matches standard
        if (!verify_scatter_correctness(data, 55)) {
            printf("  FAILED! Scatter-gather mismatch\n");
            failures++;
        } else {
            printf("  PASSED (scatter-gather matches)\n");
        }
    }

    // Test 4: Exactly 56 bytes (boundary: needs two padding blocks)
    {
        printf("\nTest 4: 56 bytes (two padding blocks boundary)\n");
        uint8_t data[56];
        memset(data, 'B', 56);
        uint8_t hash[32];
        sha256_standard(data, 56, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        if (!verify_scatter_correctness(data, 56)) {
            printf("  FAILED! Scatter-gather mismatch\n");
            failures++;
        } else {
            printf("  PASSED (scatter-gather matches)\n");
        }
    }

    // Test 5: Exactly 64 bytes (one full block)
    {
        printf("\nTest 5: 64 bytes (one full block)\n");
        uint8_t data[64];
        memset(data, 'C', 64);
        uint8_t hash[32];
        sha256_standard(data, 64, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        if (!verify_scatter_correctness(data, 64)) {
            printf("  FAILED! Scatter-gather mismatch\n");
            failures++;
        } else {
            printf("  PASSED (scatter-gather matches)\n");
        }

        // Also verify rolling vs full
        if (!verify_rolling_vs_full(data, 64)) {
            printf("  FAILED! Rolling vs Full mismatch\n");
            failures++;
        }
    }

    // Test 6: 128 bytes (two full blocks + padding)
    {
        printf("\nTest 6: 128 bytes (two full blocks + padding)\n");
        uint8_t data[128];
        for (int i = 0; i < 128; i++) {
            data[i] = (uint8_t)i;
        }
        uint8_t hash[32];
        sha256_standard(data, 128, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        if (!verify_scatter_correctness(data, 128)) {
            printf("  FAILED! Scatter-gather mismatch\n");
            failures++;
        } else {
            printf("  PASSED (scatter-gather matches)\n");
        }

        // Verify rolling vs full
        if (!verify_rolling_vs_full(data, 128)) {
            printf("  FAILED! Rolling vs Full mismatch\n");
            failures++;
        }
    }

    // Test 7: Large data (simulating SA-like sizes)
    {
        printf("\nTest 7: Large data (1024 bytes)\n");
        uint8_t data[1024];
        for (int i = 0; i < 1024; i++) {
            data[i] = (uint8_t)(i * 7 + 13);  // Some pattern
        }
        uint8_t hash[32];
        sha256_standard(data, 1024, hash);
        hash_to_hex(hash, hex);
        printf("  Result: %s\n", hex);

        if (!verify_scatter_correctness(data, 1024)) {
            printf("  FAILED! Scatter-gather mismatch\n");
            failures++;
        } else {
            printf("  PASSED (scatter-gather matches)\n");
        }
    }

    // Test 8: Test C++ function wrapper
    {
        printf("\nTest 8: C++ std::function wrapper\n");
        const uint8_t data[] = {'t', 'e', 's', 't'};
        uint8_t hash_c[32];
        uint8_t hash_cpp[32];

        // C function pointer version
        ContiguousContext ctx{data, 4};
        sha256_scatter(get_byte_contiguous, &ctx, 4, hash_c);

        // C++ std::function version
        GetByteFunc func = [&data](size_t index) -> uint8_t {
            return index < 4 ? data[index] : 0;
        };
        sha256_scatter_func(func, 4, hash_cpp);

        if (!hashes_equal(hash_c, hash_cpp)) {
            printf("  FAILED! C++ wrapper produces different result\n");
            failures++;
        } else {
            printf("  PASSED\n");
        }
    }

    // Test 9: Scattered access pattern (simulate SA decompression)
    {
        printf("\nTest 9: Scattered access pattern (interleaved data)\n");

        // Simulate data scattered across two buffers
        uint8_t buffer_a[64] = {0};
        uint8_t buffer_b[64] = {0};
        for (int i = 0; i < 64; i++) {
            buffer_a[i] = (uint8_t)(i * 2);      // Even indices
            buffer_b[i] = (uint8_t)(i * 2 + 1);  // Odd indices
        }

        // Create interleaved "virtual" data
        uint8_t interleaved[128];
        for (int i = 0; i < 128; i++) {
            if (i % 2 == 0) {
                interleaved[i] = buffer_a[i / 2];
            } else {
                interleaved[i] = buffer_b[i / 2];
            }
        }

        // Hash with standard (contiguous)
        uint8_t hash_standard[32];
        sha256_standard(interleaved, 128, hash_standard);

        // Hash with scatter (interleaved access)
        struct InterleavedContext {
            uint8_t* buf_a;
            uint8_t* buf_b;
        };
        InterleavedContext ictx{buffer_a, buffer_b};

        auto interleaved_getter = [](size_t index, void* ctx) -> uint8_t {
            InterleavedContext* ic = static_cast<InterleavedContext*>(ctx);
            if (index % 2 == 0) {
                return ic->buf_a[index / 2];
            } else {
                return ic->buf_b[index / 2];
            }
        };

        uint8_t hash_scatter[32];
        sha256_scatter(interleaved_getter, &ictx, 128, hash_scatter);

        if (!hashes_equal(hash_standard, hash_scatter)) {
            printf("  FAILED! Scattered access produces different result\n");
            hash_to_hex(hash_standard, hex);
            printf("  Standard: %s\n", hex);
            hash_to_hex(hash_scatter, hex);
            printf("  Scatter:  %s\n", hex);
            failures++;
        } else {
            printf("  PASSED\n");
        }
    }

    // Summary
    printf("\n=== Test Summary ===\n");
    if (failures == 0) {
        printf("All tests passed!\n");
    } else {
        printf("%d test(s) FAILED\n", failures);
    }

    return failures;
}

} // namespace sha256_scatter

// ============================================================================
// Standalone test program (when compiled with -DTEST_SHA256_SCATTER)
// ============================================================================

#ifdef TEST_SHA256_SCATTER
int main() {
    return sha256_scatter::run_correctness_tests();
}
#endif
