// test_sha256_x2.cpp - two-lane SHA-NI kernel vs OpenSSL, over the length and
// alignment shapes the suffix-array hashing path actually produces.

#include "sha256_x2.h"

#include <openssl/sha.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static uint64_t rng = 0x9e3779b97f4a7c15ULL;

static uint64_t next_rand()
{
    rng ^= rng >> 12;
    rng ^= rng << 25;
    rng ^= rng >> 27;
    return rng * 0x2545f4914f6cdd1dULL;
}

static void report(const char *label, const char *lane, size_t la, size_t lb,
                   const uint8_t *got, const uint8_t *want)
{
    int i = 0;
    while (i < 31 && got[i] == want[i]) i++;
    failures++;
    std::printf("FAIL %s lane %s: lens=(%zu,%zu) digest[%d] got %02x want %02x\n",
                label, lane, la, lb, i, got[i], want[i]);
}

/* One pair, checked in both lane orders. A kernel that crossed its two lanes
 * still matches the oracle if only one ordering is ever tried. */
static void check(const char *label, const uint8_t *a, size_t la,
                  const uint8_t *b, size_t lb)
{
    uint8_t want_a[32], want_b[32], got_a[32], got_b[32];
    SHA256(a, la, want_a);
    SHA256(b, lb, want_b);

    dluna_sha256_x2(a, la, got_a, b, lb, got_b);
    if (std::memcmp(got_a, want_a, 32)) report(label, "A", la, lb, got_a, want_a);
    if (std::memcmp(got_b, want_b, 32)) report(label, "B", la, lb, got_b, want_b);

    dluna_sha256_x2(b, lb, got_a, a, la, got_b);
    if (std::memcmp(got_a, want_b, 32)) report(label, "A swapped", lb, la, got_a, want_b);
    if (std::memcmp(got_b, want_a, 32)) report(label, "B swapped", lb, la, got_b, want_a);
}

/* Published vectors, so that a defect shared by the kernel and the oracle
 * cannot quietly make every comparison above vacuous. The 56-byte case is the
 * length at which padding spills into a second block. */
static void check_kat()
{
    static const struct { const char *msg; const char *hex; } kat[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    for (const auto &k : kat) {
        const size_t n = std::strlen(k.msg);
        const uint8_t *m = reinterpret_cast<const uint8_t *>(k.msg);
        uint8_t da[32], db[32];
        char hex[65];
        dluna_sha256_x2(m, n, da, m, n, db);
        for (int i = 0; i < 32; i++) std::snprintf(hex + i * 2, 3, "%02x", da[i]);
        if (std::strcmp(hex, k.hex) != 0) {
            failures++;
            std::printf("FAIL kat len=%zu: got %s want %s\n", n, hex, k.hex);
        }
        if (std::memcmp(da, db, 32) != 0) {
            failures++;
            std::printf("FAIL kat len=%zu: lanes disagree on identical input\n", n);
        }
    }
}

int main()
{
    if (!dluna_sha256_x2_available()) {
        std::printf("sha256_x2: SKIP (no SHA-NI)\n");
        return 0;
    }

    check_kat();

    /* Oversized so the two lanes can start at different offsets: the message
     * pointer's alignment is a live variable for the 16-byte loads. */
    const size_t maxlen = 300000;
    std::vector<uint8_t> buf(maxlen + 8);
    for (size_t i = 0; i < buf.size(); i += 8) {
        const uint64_t v = next_rand();
        const size_t n = buf.size() - i < 8 ? buf.size() - i : 8;
        std::memcpy(&buf[i], &v, n);
    }
    const uint8_t *base = buf.data();

    /* Block and padding boundaries, each lane independently. */
    static const size_t bounds[] = {0, 1, 55, 56, 57, 63, 64, 65,
                                    119, 120, 121, 127, 128};
    const size_t nb = sizeof(bounds) / sizeof(bounds[0]);
    for (size_t i = 0; i < nb; i++)
        for (size_t j = 0; j < nb; j++)
            check("boundary", base + (i & 3), bounds[i], base + 4 + (j & 3), bounds[j]);

    /* Shapes the live path produces: SA byte length is data_len*4 with data_len
     * up to ~70911, so the lanes differ by a few bytes rather than a few
     * percent. The zero-length lane is the one that idles for every block. */
    check("sa_idle_lane", base, 0, base, 300000);
    check("sa_275k", base, 275354, base + 1, 275352);
    check("sa_283k", base + 2, 283644, base + 3, 283640);

    /* Identical inputs must produce identical digests. */
    {
        uint8_t da[32], db[32];
        dluna_sha256_x2(base, 275354, da, base, 275354, db);
        if (std::memcmp(da, db, 32) != 0) {
            failures++;
            std::printf("FAIL identical_inputs: lanes disagree on the same message\n");
        }
        check("identical", base, 275354, base, 275354);
    }

    for (int c = 0; c < 200; c++) {
        const size_t oa = static_cast<size_t>(next_rand() % 4);
        const size_t ob = static_cast<size_t>(next_rand() % 4);
        const size_t la = static_cast<size_t>(next_rand() % (maxlen + 1));
        const size_t lb = static_cast<size_t>(next_rand() % (maxlen + 1));
        check("random", base + oa, la, base + ob, lb);
    }

    if (failures) {
        std::printf("sha256_x2: %d FAILURES\n", failures);
        return 1;
    }
    std::printf("sha256_x2: all tests passed\n");
    return 0;
}
