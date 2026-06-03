/**
 * SHA-256 Override: SHA-NI hardware-accelerated replacement for OpenSSL SHA256
 *
 * Replaces OpenSSL's SHA256_Init/Update/Final with Intel SHA-NI instructions
 * for ~3-5x faster hashing. Uses --wrap linker flags to redirect all calls.
 *
 * Ported from DirtyBird miner's sha256_override.cpp (stripped of telemetry).
 */

#include <cstdint>
#include <cstring>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <cpuid.h>
#endif

#include <openssl/sha.h>

static const uint32_t SHA256_H0[8] = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

alignas(16) static const uint32_t SHA256_K_NI[64] = {
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
// SHA-NI detection and block processing
// ============================================================================

#if defined(__x86_64__) || defined(_M_X64)

static bool g_sha_ni_available = false;
static bool g_sha_ni_checked = false;

static bool check_sha_ni() {
    unsigned int eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        return (ebx & (1 << 29)) != 0;
    }
    return false;
}

__attribute__((target("sha,sse4.1,ssse3")))
static inline void sha256_state_to_ni(const uint32_t state[8], __m128i& state0, __m128i& state1) {
    state0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state));
    state1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(state + 4));
    __m128i tmp = _mm_shuffle_epi32(state0, 0xB1);
    __m128i tmp2 = _mm_shuffle_epi32(state1, 0x1B);
    state0 = _mm_alignr_epi8(tmp, tmp2, 8);
    state1 = _mm_blend_epi16(tmp2, tmp, 0xF0);
}

__attribute__((target("sha,sse4.1,ssse3")))
static inline void sha256_state_from_ni(uint32_t state[8], __m128i state0, __m128i state1) {
    __m128i tmp = _mm_shuffle_epi32(state0, 0x1B);
    state1 = _mm_shuffle_epi32(state1, 0xB1);
    state0 = _mm_blend_epi16(tmp, state1, 0xF0);
    state1 = _mm_alignr_epi8(state1, tmp, 8);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state), state0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(state + 4), state1);
}

__attribute__((target("sha,sse4.1,ssse3")))
static inline void sha256_block_ni_core(__m128i& state0, __m128i& state1,
                                         const uint8_t* block, __m128i shuf_mask) {
    __m128i abef_save = state0;
    __m128i cdgh_save = state1;
    const __m128i* k_ptr = reinterpret_cast<const __m128i*>(SHA256_K_NI);

    __m128i msg0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)block), shuf_mask);
    __m128i msg1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(block+16)), shuf_mask);
    __m128i msg2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(block+32)), shuf_mask);
    __m128i msg3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(block+48)), shuf_mask);
    __m128i msg_tmp;

    #define SHA_ROUND(r, m_cur, m_next, m_prev2, m_prev3) \
        msg_tmp = _mm_add_epi32(m_cur, _mm_loadu_si128(k_ptr + (r))); \
        state1 = _mm_sha256rnds2_epu32(state1, state0, msg_tmp); \
        msg_tmp = _mm_shuffle_epi32(msg_tmp, 0x0E); \
        state0 = _mm_sha256rnds2_epu32(state0, state1, msg_tmp);

    // Rounds 0-3
    SHA_ROUND(0, msg0, msg1, msg2, msg3);
    // Rounds 4-7
    SHA_ROUND(1, msg1, msg2, msg3, msg0);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    // Rounds 8-11
    SHA_ROUND(2, msg2, msg3, msg0, msg1);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    // Rounds 12-15
    SHA_ROUND(3, msg3, msg0, msg1, msg2);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);

    // Rounds 16-19
    SHA_ROUND(4, msg0, msg1, msg2, msg3);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    // Rounds 20-23
    SHA_ROUND(5, msg1, msg2, msg3, msg0);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    // Rounds 24-27
    SHA_ROUND(6, msg2, msg3, msg0, msg1);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    // Rounds 28-31
    SHA_ROUND(7, msg3, msg0, msg1, msg2);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    // Rounds 32-35
    SHA_ROUND(8, msg0, msg1, msg2, msg3);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    // Rounds 36-39
    SHA_ROUND(9, msg1, msg2, msg3, msg0);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    msg0 = _mm_sha256msg1_epu32(msg0, msg1);
    // Rounds 40-43
    SHA_ROUND(10, msg2, msg3, msg0, msg1);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    msg1 = _mm_sha256msg1_epu32(msg1, msg2);
    // Rounds 44-47
    SHA_ROUND(11, msg3, msg0, msg1, msg2);
    msg0 = _mm_add_epi32(msg0, _mm_alignr_epi8(msg3, msg2, 4));
    msg0 = _mm_sha256msg2_epu32(msg0, msg3);
    msg2 = _mm_sha256msg1_epu32(msg2, msg3);
    // Rounds 48-51
    SHA_ROUND(12, msg0, msg1, msg2, msg3);
    msg1 = _mm_add_epi32(msg1, _mm_alignr_epi8(msg0, msg3, 4));
    msg1 = _mm_sha256msg2_epu32(msg1, msg0);
    msg3 = _mm_sha256msg1_epu32(msg3, msg0);
    // Rounds 52-55
    SHA_ROUND(13, msg1, msg2, msg3, msg0);
    msg2 = _mm_add_epi32(msg2, _mm_alignr_epi8(msg1, msg0, 4));
    msg2 = _mm_sha256msg2_epu32(msg2, msg1);
    // Rounds 56-59
    SHA_ROUND(14, msg2, msg3, msg0, msg1);
    msg3 = _mm_add_epi32(msg3, _mm_alignr_epi8(msg2, msg1, 4));
    msg3 = _mm_sha256msg2_epu32(msg3, msg2);
    // Rounds 60-63
    SHA_ROUND(15, msg3, msg0, msg1, msg2);

    #undef SHA_ROUND

    state0 = _mm_add_epi32(state0, abef_save);
    state1 = _mm_add_epi32(state1, cdgh_save);
}

__attribute__((target("sha,sse4.1,ssse3")))
static void sha256_blocks_ni(uint32_t state[8], const uint8_t* blocks, size_t block_count) {
    if (block_count == 0) return;
    const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i state0, state1;
    sha256_state_to_ni(state, state0, state1);
    for (size_t i = 0; i < block_count; i++) {
        if (i + 8 < block_count)
            _mm_prefetch((const char*)(blocks + (i+8)*64), _MM_HINT_T0);
        sha256_block_ni_core(state0, state1, blocks + i*64, shuf_mask);
    }
    sha256_state_from_ni(state, state0, state1);
}

__attribute__((target("sha,sse4.1,ssse3")))
static void sha256_block_ni(uint32_t state[8], const uint8_t* block) {
    const __m128i shuf_mask = _mm_set_epi64x(0x0c0d0e0f08090a0bLL, 0x0405060700010203LL);
    __m128i state0, state1;
    sha256_state_to_ni(state, state0, state1);
    sha256_block_ni_core(state0, state1, block, shuf_mask);
    sha256_state_from_ni(state, state0, state1);
}

#endif // x86_64

// ============================================================================
// Software fallback
// ============================================================================

static inline uint32_t rotr32(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

static void sha256_block_soft(uint32_t state[8], const uint8_t* block) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)block[i*4]<<24)|((uint32_t)block[i*4+1]<<16)|((uint32_t)block[i*4+2]<<8)|(uint32_t)block[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotr32(w[i-15],7)^rotr32(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1 = rotr32(w[i-2],17)^rotr32(w[i-2],19)^(w[i-2]>>10);
        w[i] = w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=state[0],b=state[1],c=state[2],d=state[3];
    uint32_t e=state[4],f=state[5],g=state[6],h=state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1=rotr32(e,6)^rotr32(e,11)^rotr32(e,25), ch=(e&f)^(~e&g);
        uint32_t t1=h+S1+ch+SHA256_K_NI[i]+w[i];
        uint32_t S0=rotr32(a,2)^rotr32(a,13)^rotr32(a,22), maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    state[0]+=a;state[1]+=b;state[2]+=c;state[3]+=d;
    state[4]+=e;state[5]+=f;state[6]+=g;state[7]+=h;
}

// ============================================================================
// Dispatch
// ============================================================================

static void sha256_process_block(uint32_t state[8], const uint8_t* block) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!g_sha_ni_checked) { g_sha_ni_available = check_sha_ni(); g_sha_ni_checked = true; }
    if (g_sha_ni_available) { sha256_block_ni(state, block); return; }
#endif
    sha256_block_soft(state, block);
}

static void sha256_process_blocks(uint32_t state[8], const uint8_t* blocks, size_t n) {
#if defined(__x86_64__) || defined(_M_X64)
    if (!g_sha_ni_checked) { g_sha_ni_available = check_sha_ni(); g_sha_ni_checked = true; }
    if (g_sha_ni_available) { sha256_blocks_ni(state, blocks, n); return; }
#endif
    for (size_t i = 0; i < n; i++) sha256_block_soft(state, blocks + i*64);
}

// ============================================================================
// OpenSSL-compatible API via --wrap linker flags
// ============================================================================

#define CTX_BUF(c) ((uint8_t*)(c)->data)

extern "C" {

int __wrap_SHA256_Init(SHA256_CTX *c) {
    memcpy(c->h, SHA256_H0, sizeof(SHA256_H0));
    c->Nl = 0; c->Nh = 0; c->num = 0;
    c->md_len = SHA256_DIGEST_LENGTH;
    return 1;
}

int __wrap_SHA256_Update(SHA256_CTX *c, const void *data_, size_t len) {
    const uint8_t *data = (const uint8_t*)data_;
    uint8_t *buf = CTX_BUF(c);
    uint32_t l = c->Nl + (uint32_t)(len << 3);
    if (l < c->Nl) c->Nh++;
    c->Nh += (uint32_t)(len >> 29);
    c->Nl = l;

    if (c->num > 0) {
        uint32_t need = 64 - c->num;
        if (len < need) { memcpy(buf + c->num, data, len); c->num += (uint32_t)len; return 1; }
        memcpy(buf + c->num, data, need);
        sha256_process_block(c->h, buf);
        data += need; len -= need; c->num = 0;
    }
    if (len >= 64) {
        size_t full = len / 64;
        sha256_process_blocks(c->h, data, full);
        size_t consumed = full * 64;
        data += consumed; len -= consumed;
    }
    if (len > 0) { memcpy(buf, data, len); c->num = (uint32_t)len; }
    return 1;
}

int __wrap_SHA256_Final(unsigned char *md, SHA256_CTX *c) {
    uint8_t *buf = CTX_BUF(c);
    uint8_t *p = buf + c->num;
    *p++ = 0x80;
    uint32_t pad_len = 64 - c->num - 1;
    if (pad_len < 8) {
        memset(p, 0, pad_len);
        sha256_process_block(c->h, buf);
        memset(buf, 0, 56);
    } else {
        memset(p, 0, pad_len - 8);
    }
    buf[56]=(uint8_t)(c->Nh>>24); buf[57]=(uint8_t)(c->Nh>>16);
    buf[58]=(uint8_t)(c->Nh>>8);  buf[59]=(uint8_t)(c->Nh);
    buf[60]=(uint8_t)(c->Nl>>24); buf[61]=(uint8_t)(c->Nl>>16);
    buf[62]=(uint8_t)(c->Nl>>8);  buf[63]=(uint8_t)(c->Nl);
    sha256_process_block(c->h, buf);
    for (int i = 0; i < 8; i++) {
        md[i*4]=(uint8_t)(c->h[i]>>24); md[i*4+1]=(uint8_t)(c->h[i]>>16);
        md[i*4+2]=(uint8_t)(c->h[i]>>8); md[i*4+3]=(uint8_t)(c->h[i]);
    }
    return 1;
}

} // extern "C"

#undef CTX_BUF
