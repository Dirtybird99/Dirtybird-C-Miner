// test_cap_format.cpp - round-trip tests for the .cap binary schema.
// Exit code 0 on all-pass, non-zero on any failure.

#include "cap_format.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace deroluna::replay;

static int g_failed = 0;
#define CHECK(cond) do { if (!(cond)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
    g_failed++; } } while (0)

static fs::path tmpfile(const char* tag) {
    auto p = fs::temp_directory_path() /
             (std::string("dlcap_test_") + tag + "_" +
              std::to_string(std::rand()) + ".cap");
    return p;
}

static void test_roundtrip_one_stage_small() {
    auto path = tmpfile("small");
    std::vector<uint8_t> in_buf  = {1, 2, 3, 4};
    std::vector<uint8_t> out_buf = {9, 8, 7};
    {
        CaptureWriter w;
        CHECK(w.open(path.string().c_str(), 42));
        CHECK(w.write_stage(StageId::Salsa20Init,
                            in_buf.data(),  in_buf.size(),
                            out_buf.data(), out_buf.size()));
        CHECK(w.close());
    }
    {
        CaptureReader r;
        CHECK(r.open(path.string().c_str()));
        CHECK(r.hash_seq() == 42);
        CHECK(r.stage_count() == 1);
        StageRecord rec;
        CHECK(r.read_stage(0, &rec));
        CHECK(rec.stage_id == StageId::Salsa20Init);
        CHECK(rec.in_len == in_buf.size());
        CHECK(rec.out_len == out_buf.size());
        CHECK(std::memcmp(rec.in_bytes, in_buf.data(), in_buf.size()) == 0);
        CHECK(std::memcmp(rec.out_bytes, out_buf.data(), out_buf.size()) == 0);
    }
    fs::remove(path);
}

static void test_roundtrip_six_stages_large() {
    auto path = tmpfile("large");
    std::vector<std::vector<uint8_t>> ins(6), outs(6);
    for (int s = 0; s < 6; ++s) {
        ins[s].resize(1 + s * 17);
        outs[s].resize(128 * 1024 + s);
        for (size_t i = 0; i < ins[s].size();  ++i) ins[s][i]  = (uint8_t)(s * 31 + i);
        for (size_t i = 0; i < outs[s].size(); ++i) outs[s][i] = (uint8_t)(s * 53 + i);
    }
    {
        CaptureWriter w;
        CHECK(w.open(path.string().c_str(), 7));
        for (int s = 0; s < 6; ++s) {
            CHECK(w.write_stage(static_cast<StageId>(s + 1),
                                ins[s].data(),  ins[s].size(),
                                outs[s].data(), outs[s].size()));
        }
        CHECK(w.close());
    }
    {
        CaptureReader r;
        CHECK(r.open(path.string().c_str()));
        CHECK(r.stage_count() == 6);
        for (int s = 0; s < 6; ++s) {
            StageRecord rec;
            CHECK(r.read_stage(s, &rec));
            CHECK(rec.stage_id == static_cast<StageId>(s + 1));
            CHECK(rec.in_len  == ins[s].size());
            CHECK(rec.out_len == outs[s].size());
            CHECK(std::memcmp(rec.in_bytes,  ins[s].data(),  ins[s].size())  == 0);
            CHECK(std::memcmp(rec.out_bytes, outs[s].data(), outs[s].size()) == 0);
        }
    }
    fs::remove(path);
}

static void test_zero_length_buffers() {
    auto path = tmpfile("zero");
    {
        CaptureWriter w;
        CHECK(w.open(path.string().c_str(), 0));
        CHECK(w.write_stage(StageId::Sha256OfSa, nullptr, 0, nullptr, 0));
        CHECK(w.close());
    }
    {
        CaptureReader r;
        CHECK(r.open(path.string().c_str()));
        StageRecord rec;
        CHECK(r.read_stage(0, &rec));
        CHECK(rec.in_len == 0);
        CHECK(rec.out_len == 0);
    }
    fs::remove(path);
}

static void test_corrupted_magic() {
    auto path = tmpfile("badmagic");
    {
        std::FILE* f = std::fopen(path.string().c_str(), "wb");
        const char garbage[16] = "NOTMAGIC";
        std::fwrite(garbage, 1, 16, f);
        std::fclose(f);
    }
    CaptureReader r;
    CHECK(!r.open(path.string().c_str()));
    fs::remove(path);
}

static void test_truncated_mid_stage() {
    auto path = tmpfile("truncated");
    {
        CaptureWriter w;
        std::vector<uint8_t> data(1024, 0xAB);
        CHECK(w.open(path.string().c_str(), 1));
        CHECK(w.write_stage(StageId::Salsa20Init,
                            data.data(), data.size(), data.data(), data.size()));
        CHECK(w.close());
    }
    auto sz = fs::file_size(path);
    fs::resize_file(path, sz - 100);
    CaptureReader r;
    CHECK(r.open(path.string().c_str()));
    StageRecord rec;
    CHECK(!r.read_stage(0, &rec));
    fs::remove(path);
}

int main() {
    test_roundtrip_one_stage_small();
    test_roundtrip_six_stages_large();
    test_zero_length_buffers();
    test_corrupted_magic();
    test_truncated_mid_stage();
    if (g_failed) {
        std::fprintf(stderr, "%d test failures\n", g_failed);
        return 1;
    }
    std::printf("test_cap_format: all OK\n");
    return 0;
}
