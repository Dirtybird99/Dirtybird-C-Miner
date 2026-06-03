// test_stage5_descriptor.cpp - focused checks for descriptor-backed Stage 5 SA.

#include "../../include/dluna_v114.h"

#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include <openssl/sha.h>

static int g_failed = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_failed++; } } while (0)

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

static void write_u32_le(std::vector<uint8_t>& out, uint32_t v) {
    out.push_back(static_cast<uint8_t>(v));
    out.push_back(static_cast<uint8_t>(v >> 8));
    out.push_back(static_cast<uint8_t>(v >> 16));
    out.push_back(static_cast<uint8_t>(v >> 24));
}

static uint32_t key3(const std::vector<uint8_t>& data, uint32_t pos) {
    uint32_t v = 0;
    if (pos + 0 < data.size()) v |= static_cast<uint32_t>(data[pos + 0]);
    if (pos + 1 < data.size()) v |= static_cast<uint32_t>(data[pos + 1]) << 8;
    if (pos + 2 < data.size()) v |= static_cast<uint32_t>(data[pos + 2]) << 16;
    return v;
}

static std::vector<uint8_t> sha256_bytes(const std::vector<uint8_t>& data) {
    std::vector<uint8_t> digest(32);
    SHA256(data.data(), data.size(), digest.data());
    return digest;
}

static void append_desc(std::vector<uint8_t>& desc, uint32_t key,
                        uint32_t arena_index, uint32_t count) {
    write_u32_le(desc, key);
    write_u32_le(desc, (count << 17) + arena_index);
}

static void append_literal_desc(std::vector<uint8_t>& desc, uint32_t key,
                                uint32_t pos) {
    write_u32_le(desc, key);
    write_u32_le(desc, pos);
}

static std::vector<uint8_t> make_stage5_input(const std::vector<uint8_t>& data,
                                              const std::vector<uint8_t>& arena,
                                              const std::vector<uint8_t>& desc,
                                              uint32_t logical_len) {
    Stage5InputHeader h{};
    const char magic[8] = {'D', 'L', 'S', '5', 'I', 'N', 0, 0};
    std::memcpy(h.magic, magic, sizeof(h.magic));
    h.version = 1;
    h.logical_len = logical_len;
    h.data_len = static_cast<uint32_t>(data.size());
    h.int_len = static_cast<uint32_t>(arena.size());
    h.desc_len = static_cast<uint32_t>(desc.size());

    std::vector<uint8_t> out(sizeof(h) + data.size() + arena.size() + desc.size());
    uint8_t* p = out.data();
    std::memcpy(p, &h, sizeof(h));
    p += sizeof(h);
    std::memcpy(p, data.data(), data.size());
    p += data.size();
    std::memcpy(p, arena.data(), arena.size());
    p += arena.size();
    std::memcpy(p, desc.data(), desc.size());
    return out;
}

static std::vector<uint8_t> expected_sa_bytes(const std::vector<uint32_t>& sa) {
    std::vector<uint8_t> out;
    out.reserve(sa.size() * 4u);
    for (uint32_t pos : sa) {
        write_u32_le(out, pos);
    }
    return out;
}

static bool run_descriptor(const std::vector<uint8_t>& input,
                           std::vector<uint8_t>* actual) {
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_sa_build_with_mode(
        input.data(), input.size(), actual->data(), actual->size(), &out_len,
        deroluna::stages::v114::Stage5SaBuildMode::DescriptorArena);
    if (ok) {
        actual->resize(out_len);
    }
    return ok;
}

static bool run_libsais(const std::vector<uint8_t>& input,
                        std::vector<uint8_t>* actual) {
    size_t out_len = 0;
    bool ok = deroluna::stages::v114::stage_v114_sa_build_with_mode(
        input.data(), input.size(), actual->data(), actual->size(), &out_len,
        deroluna::stages::v114::Stage5SaBuildMode::Libsais);
    if (ok) {
        actual->resize(out_len);
    }
    return ok;
}

static void test_raw_encode_with_arena_matches_packed_descriptor_path() {
    const uint32_t logical_len = 512;
    std::vector<uint8_t> data(logical_len + 3, 0);
    for (uint32_t i = 0; i < logical_len; ++i) {
        data[i] = static_cast<uint8_t>((i * 17u + (i >> 3) * 29u + 7u) & 0xffu);
    }
    std::vector<uint8_t> flags = {0, 0, 0};

    std::vector<uint8_t> desc(logical_len * 8u);
    std::vector<uint8_t> arena(logical_len * 4u);
    size_t desc_len = 0;
    size_t arena_len = 0;
    bool encode_ok = deroluna::stages::v114::stage_v114_encode_with_arena_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        desc.data(), desc.size(), &desc_len,
        arena.data(), arena.size(), &arena_len);

    CHECK(encode_ok);
    CHECK(arena_len == logical_len * 4u);
    desc.resize(desc_len);
    arena.resize(arena_len);

    std::vector<uint8_t> packed = make_stage5_input(data, arena, desc, logical_len);
    std::vector<uint8_t> from_packed(logical_len * 4u);
    std::vector<uint8_t> from_raw(logical_len * 4u);
    std::vector<uint8_t> from_trusted(logical_len * 4u);
    std::vector<uint8_t> from_libsais(logical_len * 4u);

    bool packed_ok = run_descriptor(packed, &from_packed);
    size_t raw_len = 0;
    bool raw_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        arena.data(), static_cast<uint32_t>(arena.size()),
        desc.data(), static_cast<uint32_t>(desc.size()),
        from_raw.data(), from_raw.size(), &raw_len);
    if (raw_ok) {
        from_raw.resize(raw_len);
    }
    size_t trusted_len = 0;
    bool trusted_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_trusted_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        arena.data(), static_cast<uint32_t>(arena.size()),
        desc.data(), static_cast<uint32_t>(desc.size()),
        from_trusted.data(), from_trusted.size(), &trusted_len);
    if (trusted_ok) {
        from_trusted.resize(trusted_len);
    }
    bool libsais_ok = run_libsais(packed, &from_libsais);

    CHECK(packed_ok);
    CHECK(raw_ok);
    CHECK(trusted_ok);
    CHECK(libsais_ok);
    CHECK(from_raw == from_packed);
    CHECK(from_trusted == from_packed);
    CHECK(from_raw == from_libsais);
    CHECK(from_trusted == from_libsais);
}

static void test_raw_encode_rejects_descriptor_index_overflow() {
    const uint32_t logical_len = 0x20001u;
    std::vector<uint8_t> data(logical_len + 3u, 0);
    for (uint32_t i = 0; i < logical_len; ++i) {
        data[i] = static_cast<uint8_t>((i * 31u + 11u) & 0xffu);
    }

    const uint32_t full_groups = logical_len >> 8;
    std::vector<uint8_t> flags(static_cast<size_t>(full_groups) + 1u, 1);
    flags[0] = 0;

    std::vector<uint8_t> desc(static_cast<size_t>(logical_len) * 8u);
    std::vector<uint8_t> arena(static_cast<size_t>(logical_len) * 4u);
    size_t desc_len = 0;
    size_t arena_len = 0;
    bool encode_ok = deroluna::stages::v114::stage_v114_encode_with_arena_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        desc.data(), desc.size(), &desc_len,
        arena.data(), arena.size(), &arena_len);

    CHECK(!encode_ok);
    CHECK(desc_len == 0);
    CHECK(arena_len == 0);
}

static void test_compact_raw_encode_skips_identity_arena_and_matches_libsais() {
    const uint32_t logical_len = 512;
    std::vector<uint8_t> data(logical_len + 3, 0);
    for (uint32_t i = 0; i < logical_len; ++i) {
        data[i] = static_cast<uint8_t>((i * 41u + (i >> 2) * 13u + 19u) & 0xffu);
    }
    std::vector<uint8_t> flags = {0, 0, 0};

    std::vector<uint8_t> desc(logical_len * 8u);
    std::vector<uint8_t> arena(logical_len * 4u);
    size_t desc_len = 0;
    size_t arena_len = 0;
    bool encode_ok = deroluna::stages::v114::stage_v114_encode_compact_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        desc.data(), desc.size(), &desc_len,
        arena.data(), arena.size(), &arena_len);

    CHECK(encode_ok);
    CHECK(desc_len != 0);
    const char* singleton_mode = std::getenv("DLUNA_STAGE5_COUNT1_SINGLETONS");
    if (singleton_mode && singleton_mode[0] == '1' && singleton_mode[1] == '\0') {
        CHECK(arena_len <= logical_len * 4u);
    } else {
        CHECK(arena_len < logical_len * 4u);
    }
    desc.resize(desc_len);
    arena.resize(arena_len);

    std::vector<uint8_t> packed = make_stage5_input(data, arena, desc, logical_len);
    std::vector<uint8_t> from_trusted(logical_len * 4u);
    std::vector<uint8_t> from_libsais(logical_len * 4u);

    size_t trusted_len = 0;
    bool trusted_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_trusted_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        arena.data(), static_cast<uint32_t>(arena.size()),
        desc.data(), static_cast<uint32_t>(desc.size()),
        from_trusted.data(), from_trusted.size(), &trusted_len);
    if (trusted_ok) {
        from_trusted.resize(trusted_len);
    }
    bool libsais_ok = run_libsais(packed, &from_libsais);

    CHECK(trusted_ok);
    CHECK(libsais_ok);
    CHECK(from_trusted == from_libsais);
}

static void test_compact_fused_raw_matches_compact_descriptor_and_libsais() {
    const uint32_t logical_len = 1024;
    std::vector<uint8_t> data(logical_len + 3, 0);
    for (uint32_t i = 0; i < logical_len; ++i) {
        data[i] = static_cast<uint8_t>((i * 73u + (i >> 4) * 19u + 23u) & 0xffu);
    }
    std::vector<uint8_t> flags = {1, 0, 1, 0, 0};

    std::vector<uint8_t> desc(logical_len * 8u);
    std::vector<uint8_t> arena(logical_len * 4u);
    size_t desc_len = 0;
    size_t arena_len = 0;
    bool encode_ok = deroluna::stages::v114::stage_v114_encode_compact_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        desc.data(), desc.size(), &desc_len,
        arena.data(), arena.size(), &arena_len);
    CHECK(encode_ok);
    desc.resize(desc_len);
    arena.resize(arena_len);

    std::vector<uint8_t> from_trusted(logical_len * 4u);
    size_t trusted_len = 0;
    bool trusted_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_trusted_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        arena.data(), static_cast<uint32_t>(arena.size()),
        desc.data(), static_cast<uint32_t>(desc.size()),
        from_trusted.data(), from_trusted.size(), &trusted_len);
    if (trusted_ok) {
        from_trusted.resize(trusted_len);
    }

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, logical_len);
    std::vector<uint8_t> from_libsais(logical_len * 4u);
    bool libsais_ok = run_libsais(input, &from_libsais);

    std::vector<uint8_t> from_fused(logical_len * 4u);
    size_t fused_len = 0;
    bool fused_ok = deroluna::stages::v114::stage_v114_sa_build_compact_fused_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        from_fused.data(), from_fused.size(), &fused_len);
    if (fused_ok) {
        from_fused.resize(fused_len);
    }

    CHECK(trusted_ok);
    CHECK(libsais_ok);
    CHECK(fused_ok);
    CHECK(from_fused == from_trusted);
    CHECK(from_fused == from_libsais);

    std::vector<uint8_t> fused_hash(32);
    bool hash_ok = deroluna::stages::v114::stage_v114_hash_compact_fused_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        flags.data(), static_cast<uint32_t>(flags.size()),
        fused_hash.data());
    CHECK(hash_ok);
    CHECK(fused_hash == sha256_bytes(from_fused));
}

static void test_descriptor_mode_builds_banana_suffix_array() {
    const std::vector<uint8_t> data = {'b', 'a', 'n', 'a', 'n', 'a', 0, 0};

    std::vector<uint8_t> arena;
    std::vector<uint8_t> desc;
    for (uint32_t pos : {0u, 1u, 2u, 3u, 4u, 5u}) {
        const uint32_t arena_index = static_cast<uint32_t>(arena.size() / 4u);
        write_u32_le(arena, pos);
        append_desc(desc, key3(data, pos), arena_index, 1);
    }

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 6);
    std::vector<uint8_t> actual(6 * 4);
    bool ok = run_descriptor(input, &actual);

    CHECK(ok);
    CHECK(actual == expected_sa_bytes({5, 3, 1, 0, 4, 2}));
}

static void test_descriptor_mode_expands_arena_groups_before_sorting_bucket() {
    const std::vector<uint8_t> data = {
        'a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c', 0, 0
    };

    std::vector<uint8_t> arena;
    for (uint32_t pos : {6u, 3u, 0u, 4u, 1u, 5u, 2u, 7u, 8u}) {
        write_u32_le(arena, pos);
    }

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 2), 5, 2);
    append_desc(desc, key3(data, 0), 0, 3);
    append_desc(desc, key3(data, 7), 7, 1);
    append_desc(desc, key3(data, 1), 3, 2);
    append_desc(desc, key3(data, 8), 8, 1);

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 9);
    std::vector<uint8_t> actual(9 * 4);
    bool ok = run_descriptor(input, &actual);

    CHECK(ok);
    CHECK(actual == expected_sa_bytes({6, 3, 0, 7, 4, 1, 8, 5, 2}));
}

static void test_descriptor_mode_adds_missing_zero_tail_suffixes() {
    const std::vector<uint8_t> data = {'a', 0, 0, 0};

    std::vector<uint8_t> arena;
    write_u32_le(arena, 0);

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 0), 0, 1);

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 4);
    std::vector<uint8_t> actual(4 * 4);
    bool ok = run_descriptor(input, &actual);

    CHECK(ok);
    CHECK(actual == expected_sa_bytes({3, 2, 1, 0}));
}

static void test_descriptor_mode_rejects_missing_nonzero_suffixes() {
    const std::vector<uint8_t> data = {'a', 'b', 0};

    std::vector<uint8_t> arena;
    write_u32_le(arena, 0);

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 0), 0, 1);

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 3);
    std::vector<uint8_t> actual(3 * 4);
    bool ok = run_descriptor(input, &actual);

    CHECK(!ok);
}

static void test_descriptor_mode_rejects_duplicate_positions() {
    const std::vector<uint8_t> data = {'a', 'b', 'c', 0, 0};

    std::vector<uint8_t> arena;
    write_u32_le(arena, 0);
    write_u32_le(arena, 0);
    write_u32_le(arena, 2);

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 0), 0, 2);
    append_desc(desc, key3(data, 2), 2, 1);

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 3);
    std::vector<uint8_t> actual(3 * 4);
    bool ok = run_descriptor(input, &actual);

    CHECK(!ok);
}

static void test_descriptor_mode_rejects_key_position_mismatch() {
    const std::vector<uint8_t> data = {'a', 'b', 'c', 0, 0};

    std::vector<uint8_t> arena;
    write_u32_le(arena, 0);

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 1), 0, 1);

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 1);
    std::vector<uint8_t> actual(4);
    bool ok = run_descriptor(input, &actual);

    CHECK(!ok);
}

static void test_descriptor_mode_rejects_unsorted_run_positions() {
    const std::vector<uint8_t> data = {
        'a', 'b', 'c', 'x', 'a', 'b', 'c', 'w', 0, 0, 0
    };

    std::vector<uint8_t> arena;
    write_u32_le(arena, 0);
    write_u32_le(arena, 4);
    write_u32_le(arena, 1);
    write_u32_le(arena, 2);
    write_u32_le(arena, 3);
    write_u32_le(arena, 5);
    write_u32_le(arena, 6);
    write_u32_le(arena, 7);

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 0), 0, 2);
    append_desc(desc, key3(data, 1), 2, 1);
    append_desc(desc, key3(data, 2), 3, 1);
    append_desc(desc, key3(data, 3), 4, 1);
    append_desc(desc, key3(data, 5), 5, 1);
    append_desc(desc, key3(data, 6), 6, 1);
    append_desc(desc, key3(data, 7), 7, 1);

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, 8);
    std::vector<uint8_t> actual(8 * 4);
    bool ok = run_descriptor(input, &actual);

    CHECK(!ok);
}

static void test_trusted_descriptor_sorts_literal_equal_key_group() {
    const std::vector<uint8_t> data = {
        'a', 'a', 'a', 'd', 'a', 'a', 'a', 'b', 'a', 0, 0, 0
    };
    const uint32_t logical_len = 9;

    std::vector<uint8_t> desc;
    for (uint32_t pos = 0; pos < logical_len; ++pos) {
        append_literal_desc(desc, key3(data, pos), pos);
    }

    std::vector<uint8_t> from_trusted(logical_len * 4u);
    size_t trusted_len = 0;
    bool trusted_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_trusted_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        nullptr, 0, desc.data(), static_cast<uint32_t>(desc.size()),
        from_trusted.data(), from_trusted.size(), &trusted_len);
    if (trusted_ok) {
        from_trusted.resize(trusted_len);
    }

    std::vector<uint8_t> input = make_stage5_input(data, {}, desc, logical_len);
    std::vector<uint8_t> from_libsais(logical_len * 4u);
    bool libsais_ok = run_libsais(input, &from_libsais);

    CHECK(trusted_ok);
    CHECK(libsais_ok);
    CHECK(from_trusted == from_libsais);
}

static void test_trusted_descriptor_merges_two_equal_key_runs() {
    const std::vector<uint8_t> data = {
        'a', 'b', 'c', 'd',
        'a', 'b', 'c', 'c',
        'a', 'b', 'c', 'b',
        'a', 'b', 'c', 'a',
        0, 0, 0
    };
    const uint32_t logical_len = 16;

    std::vector<uint8_t> arena;
    for (uint32_t pos : {12u, 4u, 8u, 0u}) {
        write_u32_le(arena, pos);
    }

    std::vector<uint8_t> desc;
    append_desc(desc, key3(data, 0), 0, 2);
    append_desc(desc, key3(data, 0), 2, 2);
    for (uint32_t pos = 0; pos < logical_len; ++pos) {
        if (pos == 0 || pos == 4 || pos == 8 || pos == 12) continue;
        append_literal_desc(desc, key3(data, pos), pos);
    }

    std::vector<uint8_t> from_trusted(logical_len * 4u);
    size_t trusted_len = 0;
    bool trusted_ok = deroluna::stages::v114::stage_v114_sa_build_descriptor_trusted_raw(
        data.data(), logical_len, static_cast<uint32_t>(data.size()),
        arena.data(), static_cast<uint32_t>(arena.size()),
        desc.data(), static_cast<uint32_t>(desc.size()),
        from_trusted.data(), from_trusted.size(), &trusted_len);
    if (trusted_ok) {
        from_trusted.resize(trusted_len);
    }

    std::vector<uint8_t> input = make_stage5_input(data, arena, desc, logical_len);
    std::vector<uint8_t> from_libsais(logical_len * 4u);
    bool libsais_ok = run_libsais(input, &from_libsais);

    CHECK(trusted_ok);
    CHECK(libsais_ok);
    CHECK(from_trusted == from_libsais);
}

int main() {
    test_raw_encode_with_arena_matches_packed_descriptor_path();
    test_raw_encode_rejects_descriptor_index_overflow();
    test_compact_raw_encode_skips_identity_arena_and_matches_libsais();
    test_compact_fused_raw_matches_compact_descriptor_and_libsais();
    test_descriptor_mode_builds_banana_suffix_array();
    test_descriptor_mode_expands_arena_groups_before_sorting_bucket();
    test_descriptor_mode_adds_missing_zero_tail_suffixes();
    test_descriptor_mode_rejects_missing_nonzero_suffixes();
    test_descriptor_mode_rejects_duplicate_positions();
    test_descriptor_mode_rejects_key_position_mismatch();
    test_descriptor_mode_rejects_unsorted_run_positions();
    test_trusted_descriptor_sorts_literal_equal_key_group();
    test_trusted_descriptor_merges_two_equal_key_runs();

    if (g_failed) {
        std::fprintf(stderr, "%d failures\n", g_failed);
        return 1;
    }
    std::printf("test_stage5_descriptor: all OK\n");
    return 0;
}
