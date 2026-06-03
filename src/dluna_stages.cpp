// dluna_stages.cpp - Uniform replay-stage wrappers around existing internals.

#include "dluna_stages.h"
#include "dluna.h"

#include <openssl/rc4.h>
#include <openssl/sha.h>

#include <cstring>
#include <limits>

void dluna_internal_salsa20_expand(const uint8_t* key32, uint8_t* out256);
uint32_t dluna_internal_branch_dispatch(const uint8_t* in_buf, uint32_t in_buf_len,
                                        uint8_t* out_buf, uint32_t out_buf_cap);

namespace deroluna::stages {

bool stage_salsa20_init(const uint8_t* in, size_t in_len,
                        uint8_t* out, size_t out_cap, size_t* out_len) {
    if (!in || !out || !out_len) return false;
    if ((in_len != 48 && in_len != 76) || out_cap < 256) return false;

    uint8_t sha_key[32];
    SHA256(in, in_len, sha_key);
    dluna_internal_salsa20_expand(sha_key, out);
    *out_len = 256;
    return true;
}

bool stage_rc4_ksa(const uint8_t* in, size_t in_len,
                   uint8_t* out, size_t out_cap, size_t* out_len) {
    if (!in || !out || !out_len) return false;
    if (in_len != 256 || out_cap < 256) return false;

    RC4_KEY key;
    RC4_set_key(&key, static_cast<int>(in_len), in);
    for (int i = 0; i < 256; ++i) {
        out[i] = static_cast<uint8_t>(key.data[i]);
    }
    *out_len = 256;
    return true;
}

bool stage_branch_dispatch(const uint8_t* in, size_t in_len,
                           uint8_t* out, size_t out_cap, size_t* out_len) {
    if (!in || !out || !out_len) return false;
    if (in_len < 4 || out_cap < 4) return false;

    uint32_t buf_len = 0;
    std::memcpy(&buf_len, in, sizeof(buf_len));
    if (buf_len > in_len - 4) return false;
    if (out_cap - 4 > std::numeric_limits<uint32_t>::max()) return false;

    uint32_t out_buf_len = dluna_internal_branch_dispatch(
        in + 4, buf_len, out + 4, static_cast<uint32_t>(out_cap - 4));
    if (out_buf_len == 0) return false;

    std::memcpy(out, &out_buf_len, sizeof(out_buf_len));
    *out_len = 4 + out_buf_len;
    return true;
}

bool stage_sha256_of_sa(const uint8_t* in, size_t in_len,
                        uint8_t* out, size_t out_cap, size_t* out_len) {
    if (!out || !out_len || out_cap < 32) return false;
    if (in_len != 0 && !in) return false;

    SHA256(in, in_len, out);
    *out_len = 32;
    return true;
}

}  // namespace deroluna::stages
