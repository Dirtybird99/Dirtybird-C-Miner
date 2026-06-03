// test_v114_encode.cpp - focused checks for the v1.14 encode replay stage.

#include "../../include/dluna_v114.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int g_failed = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_failed++; } } while (0)

#pragma pack(push, 1)
struct Stage4InputHeader {
    char magic[8];
    uint32_t version;
    uint32_t logical_len;
    uint32_t data_len;
    uint32_t flag_len;
    uint32_t reserved;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct Stage5InputHeader {
    char magic[8];
    uint32_t version;
    uint32_t logical_len;
    uint32_t data_len;
    uint32_t int_len;
    uint32_t desc_len;
};
#pragma pack(pop)

static std::vector<uint8_t> make_stage4_input(const std::vector<uint8_t>& flags,
                                              const std::vector<uint8_t>& data,
                                              uint32_t logical_len) {
    Stage4InputHeader h{};
    const char magic[8] = {'D', 'L', 'S', '4', 'I', 'N', 0, 0};
    std::memcpy(h.magic, magic, sizeof(h.magic));
    h.version = 1;
    h.logical_len = logical_len;
    h.data_len = static_cast<uint32_t>(data.size());
    h.flag_len = static_cast<uint32_t>(flags.size());

    std::vector<uint8_t> out(sizeof(h) + flags.size() + data.size());
    std::memcpy(out.data(), &h, sizeof(h));
    std::memcpy(out.data() + sizeof(h), flags.data(), flags.size());
    std::memcpy(out.data() + sizeof(h) + flags.size(), data.data(), data.size());
    return out;
}

static std::vector<uint8_t> make_stage5_input(const std::vector<uint8_t>& data,
                                              const std::vector<uint8_t>& int_arena,
                                              const std::vector<uint8_t>& desc,
                                              uint32_t logical_len) {
    Stage5InputHeader h{};
    const char magic[8] = {'D', 'L', 'S', '5', 'I', 'N', 0, 0};
    std::memcpy(h.magic, magic, sizeof(h.magic));
    h.version = 1;
    h.logical_len = logical_len;
    h.data_len = static_cast<uint32_t>(data.size());
    h.int_len = static_cast<uint32_t>(int_arena.size());
    h.desc_len = static_cast<uint32_t>(desc.size());

    std::vector<uint8_t> out(sizeof(h) + data.size() + int_arena.size() + desc.size());
    uint8_t* p = out.data();
    std::memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    std::memcpy(p, data.data(), data.size());
    p += data.size();
    std::memcpy(p, int_arena.data(), int_arena.size());
    p += int_arena.size();
    std::memcpy(p, desc.data(), desc.size());
    return out;
}

static void write_u32_le(std::vector<uint8_t>& out, size_t off, uint32_t v) {
    out[off + 0] = static_cast<uint8_t>(v);
    out[off + 1] = static_cast<uint8_t>(v >> 8);
    out[off + 2] = static_cast<uint8_t>(v >> 16);
    out[off + 3] = static_cast<uint8_t>(v >> 24);
}

static uint32_t read_u32_le(const std::vector<uint8_t>& in, size_t off) {
    return static_cast<uint32_t>(in[off + 0]) |
           (static_cast<uint32_t>(in[off + 1]) << 8) |
           (static_cast<uint32_t>(in[off + 2]) << 16) |
           (static_cast<uint32_t>(in[off + 3]) << 24);
}

static bool env_flag_enabled(const char* name) {
    const char* value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
}

static void test_sub_256_tail_path() {
    const uint32_t logical_len = 173;
    std::vector<uint8_t> data(logical_len + 3);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    }

    // logical_len >> 8 == 0, so v1.14 skips the full-group jump-table
    // path and uses only the final tail emitter.
    std::vector<uint8_t> flags = {1};
    std::vector<uint8_t> input = make_stage4_input(flags, data, logical_len);

    std::vector<uint8_t> expected(logical_len * 8);
    for (uint32_t i = 0; i < logical_len; ++i) {
        uint32_t key = static_cast<uint32_t>(data[i]) |
                       (static_cast<uint32_t>(data[i + 1]) << 8) |
                       (static_cast<uint32_t>(data[i + 2]) << 16);
        uint32_t packed = 0x00020000u + i;
        write_u32_le(expected, i * 8, key);
        write_u32_le(expected, i * 8 + 4, packed);
    }

    std::vector<uint8_t> actual(expected.size());
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_encode(
        input.data(), input.size(), actual.data(), actual.size(), &out_len);

    CHECK(ok);
    CHECK(out_len == expected.size());
    CHECK(actual == expected);
}

static void test_single_full_group_uses_direct_records() {
    const uint32_t logical_len = 256;
    std::vector<uint8_t> data(logical_len + 3);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = static_cast<uint8_t>((i * 19 + 5) & 0xff);
    }

    std::vector<uint8_t> flags = {0, 0};
    std::vector<uint8_t> input = make_stage4_input(flags, data, logical_len);

    std::vector<uint8_t> expected(logical_len * 8);
    for (uint32_t i = 0; i < logical_len; ++i) {
        uint32_t key = static_cast<uint32_t>(data[i]) |
                       (static_cast<uint32_t>(data[i + 1]) << 8) |
                       (static_cast<uint32_t>(data[i + 2]) << 16);
        write_u32_le(expected, i * 8, key);
        write_u32_le(expected, i * 8 + 4, 0x00020000u + i);
    }

    std::vector<uint8_t> actual(expected.size());
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_encode(
        input.data(), input.size(), actual.data(), actual.size(), &out_len);

    CHECK(ok);
    CHECK(out_len == expected.size());
    CHECK(actual == expected);
}

static void test_two_full_group_run_emits_grouped_reverse_rel_records() {
    const uint32_t logical_len = 512;
    std::vector<uint8_t> data(logical_len + 3, 0);
    std::vector<uint8_t> flags = {0, 0, 0};
    std::vector<uint8_t> input = make_stage4_input(flags, data, logical_len);

    std::vector<uint8_t> expected(256 * 8);
    for (uint32_t rel_from_end = 0; rel_from_end < 256; ++rel_from_end) {
        const uint32_t off = rel_from_end * 8;
        write_u32_le(expected, off, 0);
        write_u32_le(expected, off + 4, (2u << 17) + rel_from_end * 2u);
    }

    std::vector<uint8_t> actual(logical_len * 8);
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_encode(
        input.data(), input.size(), actual.data(), actual.size(), &out_len);

    CHECK(ok);
    CHECK(out_len == expected.size());
    actual.resize(out_len);
    CHECK(actual == expected);
}

static void test_legacy_raw_input_is_rejected() {
    std::vector<uint8_t> raw(300, 0x42);
    std::vector<uint8_t> out(300 * 8);
    size_t out_len = 123;
    bool ok = deroluna::stages::v114::stage_v114_encode(
        raw.data(), raw.size(), out.data(), out.size(), &out_len);

    CHECK(!ok);
    CHECK(out_len == 0);
}

static void test_stage5_builds_suffix_array_bytes() {
    std::vector<uint8_t> data = {'b', 'a', 'n', 'a', 'n', 'a', 0, 0, 0};
    std::vector<uint8_t> input = make_stage5_input(data, {}, {}, 6);

    const uint32_t expected_sa[] = {5, 3, 1, 0, 4, 2};
    std::vector<uint8_t> expected(sizeof(expected_sa));
    std::memcpy(expected.data(), expected_sa, sizeof(expected_sa));

    std::vector<uint8_t> actual(expected.size());
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_sa_build(
        input.data(), input.size(), actual.data(), actual.size(), &out_len);

    CHECK(ok);
    CHECK(out_len == expected.size());
    CHECK(actual == expected);
}

static void test_stage5_uses_actual_sha_fed_length() {
    std::vector<uint8_t> data = {'b', 'a', 'n', 'a', 'n', 'a', 0, 0, 0, 0};
    std::vector<uint8_t> input = make_stage5_input(data, {}, {}, 10);

    const uint32_t expected_sa[] = {9, 8, 7, 6, 5, 3, 1, 0, 4, 2};
    std::vector<uint8_t> expected(sizeof(expected_sa));
    std::memcpy(expected.data(), expected_sa, sizeof(expected_sa));

    std::vector<uint8_t> actual(expected.size());
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_sa_build(
        input.data(), input.size(), actual.data(), actual.size(), &out_len);

    CHECK(ok);
    CHECK(out_len == expected.size());
    CHECK(actual == expected);
}

static void test_compact_singletons_follow_count1_env() {
    const uint32_t logical_len = 4;
    std::vector<uint8_t> data = {
        0x10, 0x21, 0x32, 0x43, 0x54, 0x65, 0x76
    };
    std::vector<uint8_t> flags = {1};
    std::vector<uint8_t> desc(logical_len * 8u);
    std::vector<uint8_t> arena(logical_len * 4u);
    size_t desc_len = 0;
    size_t arena_len = 0;

    bool ok = deroluna::stages::v114::stage_v114_encode_compact_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        desc.data(), desc.size(), &desc_len,
        arena.data(), arena.size(), &arena_len);

    CHECK(ok);
    CHECK(desc_len == logical_len * 8u);

    const bool count1 = env_flag_enabled("DLUNA_STAGE5_COUNT1_SINGLETONS");
    CHECK(arena_len == (count1 ? logical_len * 4u : 0u));

    for (uint32_t pos = 0; pos < logical_len; ++pos) {
        const uint32_t key = static_cast<uint32_t>(data[pos]) |
                             (static_cast<uint32_t>(data[pos + 1]) << 8) |
                             (static_cast<uint32_t>(data[pos + 2]) << 16);
        CHECK(read_u32_le(desc, pos * 8u) == key);
        if (count1) {
            CHECK(read_u32_le(desc, pos * 8u + 4u) == (0x00020000u + pos));
            CHECK(read_u32_le(arena, pos * 4u) == pos);
        } else {
            CHECK(read_u32_le(desc, pos * 8u + 4u) == pos);
        }
    }
}

int main() {
    test_sub_256_tail_path();
    test_single_full_group_uses_direct_records();
    test_two_full_group_run_emits_grouped_reverse_rel_records();
    test_legacy_raw_input_is_rejected();
    test_stage5_builds_suffix_array_bytes();
    test_stage5_uses_actual_sha_fed_length();
    test_compact_singletons_follow_count1_env();

    if (g_failed) {
        std::fprintf(stderr, "%d test failures\n", g_failed);
        return 1;
    }
    std::printf("test_v114_encode: all OK\n");
    return 0;
}
