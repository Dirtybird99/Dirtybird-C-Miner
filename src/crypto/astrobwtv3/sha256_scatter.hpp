#pragma once
/**
 * Scatter-Gather SHA-256 Implementation
 *
 * This implementation accepts an index function instead of a contiguous buffer,
 * allowing on-demand byte fetching from scattered memory locations.
 *
 * Key Features:
 * 1. Index function interface: uint8_t (*get_byte)(size_t index, void* context)
 * 2. Rolling 16-word W buffer optimization (uses circular buffer for message schedule)
 * 3. Correctness verification against standard SHA-256
 *
 * The rolling W buffer optimization works because SHA-256 message schedule only
 * needs W[i-2], W[i-7], W[i-15], W[i-16] at any point. Instead of storing all 64
 * words, we use a circular buffer of 16 words and update in-place.
 */

#include <cstdint>
#include <cstddef>
#include <functional>

namespace sha256_scatter {

// SHA-256 initial hash values
extern const uint32_t H_INIT[8];

// SHA-256 round constants
extern const uint32_t K[64];

/**
 * Index function type for scatter-gather access
 *
 * @param index The byte index to fetch (0 to data_len-1)
 * @param context User-provided context pointer
 * @return The byte value at the given index
 */
typedef uint8_t (*GetByteFn)(size_t index, void* context);

/**
 * C++ function wrapper for index function
 */
using GetByteFunc = std::function<uint8_t(size_t index)>;

/**
 * SHA-256 context for streaming/incremental hashing
 */
struct Sha256Context {
    uint32_t state[8];      // Current hash state
    uint64_t total_len;     // Total bytes processed
    uint8_t buffer[64];     // Partial block buffer
    size_t buffer_len;      // Bytes in partial block buffer

    void init();
    void reset();
};

/**
 * Simple contiguous buffer context for testing
 */
struct ContiguousContext {
    const uint8_t* data;
    size_t len;
};

/**
 * Get byte from contiguous buffer (for testing/verification)
 */
uint8_t get_byte_contiguous(size_t index, void* context);

// ============================================================================
// Core SHA-256 Operations
// ============================================================================

/**
 * SHA-256 on scattered data using index function (C function pointer version)
 *
 * @param get_byte Function to fetch individual bytes
 * @param context User context passed to get_byte
 * @param data_len Total number of bytes to hash
 * @param output Output buffer for 32-byte hash
 */
void sha256_scatter(
    GetByteFn get_byte,
    void* context,
    size_t data_len,
    uint8_t* output
);

/**
 * SHA-256 on scattered data using index function (C++ std::function version)
 *
 * @param get_byte Function to fetch individual bytes
 * @param data_len Total number of bytes to hash
 * @param output Output buffer for 32-byte hash
 */
void sha256_scatter_func(
    GetByteFunc get_byte,
    size_t data_len,
    uint8_t* output
);

/**
 * SHA-256 with rolling 16-word W buffer optimization
 *
 * This version uses a circular buffer of 16 words for the message schedule,
 * reducing memory usage from 256 bytes (64 words) to 64 bytes (16 words).
 *
 * The optimization works because W[i] = gamma1(W[i-2]) + W[i-7] + gamma0(W[i-15]) + W[i-16]
 * only needs 4 previous values, and these fall within a 16-word window.
 *
 * @param get_byte Function to fetch individual bytes
 * @param context User context passed to get_byte
 * @param data_len Total number of bytes to hash
 * @param output Output buffer for 32-byte hash
 */
void sha256_scatter_rolling(
    GetByteFn get_byte,
    void* context,
    size_t data_len,
    uint8_t* output
);

/**
 * Process a single 64-byte block with rolling W buffer
 *
 * @param state Current hash state (modified in place)
 * @param get_byte Function to fetch individual bytes
 * @param context User context passed to get_byte
 * @param block_offset Starting byte offset for this block
 */
void process_block_rolling(
    uint32_t* state,
    GetByteFn get_byte,
    void* context,
    size_t block_offset
);

/**
 * Process a single 64-byte block with full W array (for comparison)
 *
 * @param state Current hash state (modified in place)
 * @param get_byte Function to fetch individual bytes
 * @param context User context passed to get_byte
 * @param block_offset Starting byte offset for this block
 */
void process_block_full(
    uint32_t* state,
    GetByteFn get_byte,
    void* context,
    size_t block_offset
);

// ============================================================================
// Standard SHA-256 for Verification
// ============================================================================

/**
 * Standard SHA-256 on contiguous buffer (for verification)
 *
 * @param data Input data buffer
 * @param data_len Length of input data
 * @param output Output buffer for 32-byte hash
 */
void sha256_standard(
    const uint8_t* data,
    size_t data_len,
    uint8_t* output
);

/**
 * Verify scatter-gather implementation against standard SHA-256
 *
 * @param data Test data buffer
 * @param data_len Length of test data
 * @return true if outputs match, false otherwise
 */
bool verify_scatter_correctness(
    const uint8_t* data,
    size_t data_len
);

/**
 * Run comprehensive correctness tests
 *
 * @return Number of failed tests (0 = all passed)
 */
int run_correctness_tests();

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Convert hash output to hex string
 */
void hash_to_hex(const uint8_t* hash, char* hex_str);

/**
 * Compare two hashes
 */
bool hashes_equal(const uint8_t* hash1, const uint8_t* hash2);

/**
 * Print hash in hex format
 */
void print_hash(const uint8_t* hash);

} // namespace sha256_scatter
