// port of cpu_moe.rs. Function names match the Rust ones; the scalar kernels and the AVX2 kernels follow the
// Rust code line by line (same block loops, same summation order). See cpu_moe.h for the deviations.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "cpu_moe.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#if defined(__AVX2__)
#include <immintrin.h>
#define SC_AVX2 1
#else
#define SC_AVX2 0
#endif

#include "tables_host.h"

namespace strata {

namespace {

// steady-clock ns (QueryPerformanceCounter on MSVC); used only by the STRATA_TRACE_WORKER counters
int64_t now_ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

using namespace strata::tables;

// ---------------------------------------------------------------- helpers

// Little-endian loads (x86 is little-endian, as the Rust `from_le_bytes` calls assume).
inline uint16_t le16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
inline uint32_t le32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t le64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// port of half::f16::to_f32 (exact widening, subnormals included).
float f16_bits(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu;
    uint32_t man = h & 0x3FFu;
    uint32_t out;
    if (exp == 0) {
        if (man == 0) {
            out = sign;
        } else {
            int e = -1;
            do {
                e++;
                man <<= 1;
            } while (!(man & 0x400u));
            man &= 0x3FFu;
            out = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7F800000u | (man << 13);
    } else {
        out = sign | ((exp + 112) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &out, 4);
    return f;
}

// port of cpu_moe.rs: d16
inline float d16(const uint8_t* b) { return f16_bits(le16(b)); }

// port of cpu_moe.rs: SIGNS. Signs of IQ2/IQ3 sub-blocks as +-1 bytes, indexed by the 7-bit sign field. Rust builds
// this table in a const block, so it is a constant here too: a function-local static would put a thread-safe
// initialization guard on every sign lookup, which made the IQ3_XXS kernel 2.7x slower (MODIFICATIONS.md, W3-E1c).
struct SignTable {
    int8_t t[128][8];
    constexpr SignTable() : t() {
        for (size_t i = 0; i < 128; i++)
            for (size_t j = 0; j < 8; j++) t[i][j] = (KSIGNS_IQ2XS[i] & (1u << j)) ? -1 : 1;
    }
};
inline constexpr SignTable kSigns{};
inline const int8_t* signs(size_t i) { return kSigns.t[i]; }

// The 8 bytes of a grid entry in memory order (the Rust `to_le_bytes`).
inline void grid_bytes8(uint64_t v, uint8_t* g) { std::memcpy(g, &v, 8); }

// port of cpu_moe.rs: dot8
inline int32_t dot8(const uint8_t* g, const int8_t* s, const int8_t* q) {
    int32_t r = 0;
    for (int j = 0; j < 8; j++) r += (int32_t)g[j] * (int32_t)s[j] * (int32_t)q[j];
    return r;
}

// port of cpu_moe.rs: dot1 (8 IQ1 grid values -1, 0, 1 times q)
inline int32_t dot1(uint64_t g, const int8_t* q) {
    int32_t r = 0;
    for (int k = 0; k < 8; k++) r += (int32_t)(int8_t)((g >> (8 * k)) & 0xFF) * (int32_t)q[k];
    return r;
}

// port of quant/mod.rs: scale_min_k4 (`q` is the 12 scale bytes)
inline void scale_min_k4(size_t j, const uint8_t* q, uint8_t& sc, uint8_t& mn) {
    if (j < 4) {
        sc = q[j] & 63;
        mn = q[j + 4] & 63;
    } else {
        sc = (uint8_t)((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4));
        mn = (uint8_t)((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// port of quant/mod.rs: q3k_scales (16 signed 6-bit scales of a Q3_K block)
void q3k_scales(const uint8_t* b, int8_t out[16]) {
    uint32_t aux[4] = {0, 0, 0, 0};
    for (int i = 0; i < 3; i++) aux[i] = le32(b + 96 + 4 * i);
    const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) out[4 * i + j] = (int8_t)((aux[i] >> (8 * j)) & 0xFF);
}

// port of cpu_moe.rs: silu (ops.rs)
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

// port of cpu_moe.rs: grouped. Sum over the 16 groups of 16 of sc(group) * (a . q).
template <class F>
int32_t grouped(const int8_t* a, const int8_t* q, F sc) {
    int32_t s = 0;
    for (int j = 0; j < 16; j++) {
        int32_t t = 0;
        for (int i = 16 * j; i < 16 * j + 16; i++) t += (int32_t)a[i] * (int32_t)q[i];
        s += sc(j) * t;
    }
    return s;
}

// port of cpu_moe.rs: `int_dot` picks the kernel per type; a kernel takes the row bytes and the Q8K blocks.
using DotFn = float (*)(const uint8_t* row, size_t n, const Q8K* xq, size_t nx);

// Scalar kernels (cpu_moe.rs: dot_*). `n` is the row length in bytes.

// port of cpu_moe.rs: dot_iq2xs
float dot_iq2xs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 74, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 74 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* sc = b + 66;
        int32_t bsum = 0;
        for (int ib = 0; ib < 8; ib++) {
            int32_t s[2] = {0, 0};
            for (int l = 0; l < 4; l++) {
                const size_t o = 2 + 2 * (4 * ib + l);
                const uint16_t q = le16(b + o);
                uint8_t g[8];
                grid_bytes8(IQ2XS_GRID[q & 511], g);
                s[l / 2] += dot8(g, signs(q >> 9), y.qs + 32 * ib + 8 * l);
            }
            bsum += s[0] * (2 * (sc[ib] & 0xF) + 1) + s[1] * (2 * (sc[ib] >> 4) + 1);
        }
        sumf += d16(b) * y.d * (float)bsum;
    }
    return 0.125f * sumf;
}

// port of cpu_moe.rs: dot_iq2xxs
float dot_iq2xxs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 66, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 66 * bi;
        const Q8K& y = xq[bi];
        int32_t bsum = 0;
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t aux = le32(b + 6 + 8 * ib);
            int32_t s = 0;
            for (int l = 0; l < 4; l++) {
                uint8_t g[8];
                grid_bytes8(IQ2XXS_GRID[b[2 + 8 * ib + l]], g);
                s += dot8(g, signs((aux >> (7 * l)) & 127), y.qs + 32 * ib + 8 * l);
            }
            bsum += s * (2 * (int32_t)(aux >> 28) + 1);
        }
        sumf += d16(b) * y.d * (float)bsum;
    }
    return 0.125f * sumf;
}

// port of cpu_moe.rs: dot_iq2s
float dot_iq2s(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 82, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 82 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* qs = b + 2;
        const uint8_t* signs_b = b + 34;
        const uint8_t* qh = b + 66;
        const uint8_t* sc = b + 74;
        int32_t bsum = 0;
        for (int ib = 0; ib < 8; ib++) {
            int32_t s[2];
            for (int h = 0; h < 2; h++) {
                int32_t acc = 0;
                for (int l = 2 * h; l < 2 * h + 2; l++) {
                    const size_t gi = (size_t)qs[4 * ib + l] | (((size_t)qh[ib] << (8 - 2 * l)) & 0x300);
                    uint8_t g[8];
                    grid_bytes8(IQ2S_GRID[gi], g);
                    int8_t sg[8];
                    for (int j = 0; j < 8; j++) sg[j] = (int8_t)(1 - 2 * ((signs_b[4 * ib + l] >> j) & 1));
                    acc += dot8(g, sg, y.qs + 32 * ib + 8 * l);
                }
                s[h] = acc;
            }
            bsum += s[0] * (2 * (sc[ib] & 0xF) + 1) + s[1] * (2 * (sc[ib] >> 4) + 1);
        }
        sumf += d16(b) * y.d * (float)bsum;
    }
    return 0.125f * sumf;
}

// port of cpu_moe.rs: dot_iq1s
float dot_iq1s(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 50, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 50 * bi;
        const Q8K& y = xq[bi];
        int32_t s = 0, s1 = 0;
        for (int ib = 0; ib < 8; ib++) {
            const size_t qh = le16(b + 34 + 2 * ib);
            const int32_t ls = 2 * (int32_t)((qh >> 12) & 7) + 1;
            int32_t acc = 0;
            for (int l = 0; l < 4; l++)
                acc += dot1(IQ1S_GRID[(size_t)b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)], y.qs + 32 * ib + 8 * l);
            s += ls * acc;
            s1 += ls * ((qh & 0x8000) ? -1 : 1) * (int32_t)((int32_t)y.bs[2 * ib] + (int32_t)y.bs[2 * ib + 1]);
        }
        sumf += d16(b) * y.d * ((float)s + 0.125f * (float)s1);
    }
    return sumf;
}

// port of cpu_moe.rs: iq1m_d (the top 4 bits of the four scale words)
float iq1m_d(const uint8_t* b) {
    auto sc = [&](int i) { return le16(b + 48 + 2 * i); };
    const uint16_t bits = (uint16_t)((sc(0) >> 12) | ((sc(1) >> 8) & 0xF0) | ((sc(2) >> 4) & 0xF00) | (sc(3) & 0xF000));
    return f16_bits(bits);
}

// port of cpu_moe.rs: iq1m_ls (scale of values 16 * (2 * ib + h)..+16 of an IQ1_M block)
int32_t iq1m_ls(const uint8_t* b, size_t ib, size_t h) {
    const uint16_t w = le16(b + 48 + ib / 2 * 2);
    return 2 * (int32_t)((w >> (6 * (ib % 2) + 3 * h)) & 7) + 1;
}

// port of cpu_moe.rs: dot_iq1m
float dot_iq1m(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 56, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 56 * bi;
        const Q8K& y = xq[bi];
        int32_t s1 = 0, s2 = 0;
        for (size_t ib = 0; ib < 8; ib++) {
            for (size_t l = 0; l < 4; l++) {
                const size_t qh = (size_t)b[32 + 2 * ib + l / 2] >> (4 * (l % 2));
                const int8_t* q = y.qs + 32 * ib + 8 * l;
                const int32_t ls = iq1m_ls(b, ib, l / 2);
                s1 += ls * dot1(IQ1S_GRID[(size_t)b[4 * ib + l] | ((qh & 7) << 8)], q);
                int32_t qs = 0;
                for (int k = 0; k < 8; k++) qs += q[k];
                s2 += ls * ((qh & 8) ? -1 : 1) * qs;
            }
        }
        sumf += iq1m_d(b) * y.d * ((float)s1 + 0.125f * (float)s2);
    }
    return sumf;
}

// port of cpu_moe.rs: dot_iq3xxs
float dot_iq3xxs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 98, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 98 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* qs = b + 2;
        const uint8_t* gas = b + 66;
        int32_t bsum = 0;
        for (int ib = 0; ib < 8; ib++) {
            const uint32_t aux = le32(gas + 4 * ib);
            int32_t s = 0;
            for (int l = 0; l < 4; l++) {
                uint8_t g[8];
                std::memcpy(g, &IQ3XXS_GRID[qs[8 * ib + 2 * l]], 4);
                std::memcpy(g + 4, &IQ3XXS_GRID[qs[8 * ib + 2 * l + 1]], 4);
                s += dot8(g, signs((aux >> (7 * l)) & 127), y.qs + 32 * ib + 8 * l);
            }
            bsum += s * (2 * (int32_t)(aux >> 28) + 1);
        }
        sumf += d16(b) * y.d * (float)bsum;
    }
    return 0.25f * sumf;
}

// port of cpu_moe.rs: dot_iq3s
float dot_iq3s(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 110, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 110 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* qs = b + 2;
        const uint8_t* qh = b + 66;
        const uint8_t* sgb = b + 74;
        const uint8_t* sc = b + 106;
        int32_t bsum = 0;
        for (int ib = 0; ib < 8; ib++) {
            int32_t s = 0;
            for (int l = 0; l < 4; l++) {
                auto gi = [&](int k, int sh) -> uint32_t {
                    return IQ3S_GRID[(size_t)qs[8 * ib + 2 * l + k] | (((size_t)qh[ib] << sh) & 256)];
                };
                uint32_t g1 = gi(0, 8 - 2 * l), g2 = gi(1, 7 - 2 * l);
                uint8_t g[8];
                std::memcpy(g, &g1, 4);
                std::memcpy(g + 4, &g2, 4);
                int8_t sg[8];
                for (int j = 0; j < 8; j++) sg[j] = ((sgb[4 * ib + l] >> j) & 1) ? -1 : 1;
                s += dot8(g, sg, y.qs + 32 * ib + 8 * l);
            }
            bsum += s * (2 * (int32_t)((sc[ib / 2] >> (4 * (ib % 2))) & 0xF) + 1);
        }
        sumf += d16(b) * y.d * (float)bsum;
    }
    return sumf;
}

// port of cpu_moe.rs: dot_iq4xs
float dot_iq4xs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 136, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 136 * bi;
        const Q8K& y = xq[bi];
        const float dd = d16(b) * y.d;
        const uint16_t sh = le16(b + 2);
        const uint8_t* sl = b + 4;
        const uint8_t* qs = b + 8;
        for (int ib = 0; ib < 8; ib++) {
            const int32_t ls = (int32_t)(((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | ((((sh >> (2 * ib)) & 3)) << 4));
            const int8_t* q8 = y.qs + 32 * ib;
            int32_t s = 0;
            for (int j = 0; j < 16; j++) {
                const uint8_t q = qs[16 * ib + j];
                s += (int32_t)q8[j] * (int32_t)KVALUES_IQ4NL[q & 0xF] + (int32_t)q8[j + 16] * (int32_t)KVALUES_IQ4NL[q >> 4];
            }
            sumf += dd * (float)(ls - 32) * (float)s;
        }
    }
    return sumf;
}

// port of cpu_moe.rs: dot_iq4nl (32 values d * KVALUES_IQ4NL[q] per block, eight blocks per Q8_K block)
float dot_iq4nl(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    float sum = 0.0f;
    const size_t nb = n / 18;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + 18 * i;
        const Q8K& y = xq[i / 8];
        const size_t o = (i % 8) * 32;
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t q = b[2 + j];
            s += (int32_t)KVALUES_IQ4NL[q & 0xF] * (int32_t)y.qs[o + j] + (int32_t)KVALUES_IQ4NL[q >> 4] * (int32_t)y.qs[o + j + 16];
        }
        sum += d16(b) * y.d * (float)s;
    }
    return sum;
}

// StrataFreet addition (not in cpu_moe.rs; the structure of dot_iq4nl): 32 values per 17-byte MXFP4 block, the
// scale e8m0_half(e) (ggml's halved E8M0) times the doubled kvalues_mxfp4 of each code, eight blocks per Q8_K block.
// The integer sums are exact; the float scale is an exact power of two, so the result is the dequantized dot.
float dot_mxfp4(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    float sum = 0.0f;
    const size_t nb = n / 17;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + 17 * i;
        const Q8K& y = xq[i / 8];
        const size_t o = (i % 8) * 32;
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            const uint8_t q = b[1 + j];
            s += (int32_t)kKvaluesMxfp4[q & 0xF] * (int32_t)y.qs[o + j] +
                 (int32_t)kKvaluesMxfp4[q >> 4] * (int32_t)y.qs[o + j + 16];
        }
        sum += e8m0_half(b[0]) * y.d * (float)s;
    }
    return sum;
}

// port of cpu_moe.rs: dot_q2k
float dot_q2k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 84, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 84 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* sc = b;
        int8_t a[256];
        for (int i = 0; i < 256; i++) a[i] = (int8_t)((b[16 + 32 * (i / 128) + i % 32] >> (2 * ((i / 32) % 4))) & 3);
        const int32_t s = grouped(a, y.qs, [&](int j) { return (int32_t)(sc[j] & 0xF); });
        int32_t m = 0;
        for (int j = 0; j < 16; j++) m += (int32_t)(sc[j] >> 4) * (int32_t)y.bs[j];
        sumf += y.d * d16(b + 80) * (float)s - y.d * d16(b + 82) * (float)m;
    }
    return sumf;
}

// port of cpu_moe.rs: dot_q3k
float dot_q3k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 110, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 110 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* hm = b;
        int8_t sc[16];
        q3k_scales(b, sc);
        int8_t a[256];
        for (int i = 0; i < 256; i++) {
            const int k = (i / 32) % 4 + 4 * (i / 128);
            a[i] = (int8_t)(((b[32 + 32 * (i / 128) + i % 32] >> (2 * (k % 4))) & 3) - (((hm[i % 32] >> k) & 1) ? 0 : 4));
        }
        sumf += d16(b + 108) * y.d * (float)grouped(a, y.qs, [&](int j) { return (int32_t)sc[j] - 32; });
    }
    return sumf;
}

// port of cpu_moe.rs: dot_q2_0 (Q2_0: 64 values (q - 1) * d with q in 0..4, four to a byte; four blocks per Q8_K)
float dot_q2_0(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    float sum = 0.0f;
    const size_t nb = n / 18;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + 18 * i;
        const Q8K& y = xq[i / 4];
        const size_t o = (i % 4) * 64;
        int32_t s = 0;
        for (size_t j = 0; j < 64; j++) s += (int32_t)((b[2 + j / 4] >> (2 * (j % 4))) & 3) * (int32_t)y.qs[o + j];
        int32_t m = 0;
        for (size_t k = 0; k < 4; k++) m += (int32_t)y.bs[o / 16 + k];
        sum += d16(b) * y.d * (float)(s - m);
    }
    return sum;
}

// port of cpu_moe.rs: trit (TQ1_0 base-3 digit n of byte v, as -1, 0, 1)
inline int trit(uint8_t v, int n) {
    static const uint8_t POW3[5] = {1, 3, 9, 27, 81};
    const uint8_t vw = (uint8_t)(v * POW3[n]);
    return (int)((((uint16_t)vw) * 3) >> 8) - 1;
}

// port of cpu_moe.rs: dot_tq1_0 (five base-3 trits per byte in qs, four in qh; values -1, 0, 1 times d)
float dot_tq1_0(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 54, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 54 * bi;
        const Q8K& y = xq[bi];
        int32_t s = 0;
        for (int n5 = 0; n5 < 5; n5++)
            for (int m = 0; m < 32; m++) s += trit(b[m], n5) * (int32_t)y.qs[32 * n5 + m];
        for (int n5 = 0; n5 < 5; n5++)
            for (int m = 0; m < 16; m++) s += trit(b[32 + m], n5) * (int32_t)y.qs[160 + 16 * n5 + m];
        for (int n4 = 0; n4 < 4; n4++)
            for (int j = 0; j < 4; j++) s += trit(b[48 + j], n4) * (int32_t)y.qs[240 + 4 * n4 + j];
        sumf += d16(b + 52) * y.d * (float)s;
    }
    return sumf;
}

// port of cpu_moe.rs: dot_tq2_0 (64 bytes of 2-bit codes, code - 1 in -1..2, then f16 d; 256 values a block)
float dot_tq2_0(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 66, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 66 * bi;
        const Q8K& y = xq[bi];
        int32_t s = 0;
        for (int j = 0; j < 2; j++)
            for (int l = 0; l < 4; l++)
                for (int m = 0; m < 32; m++) {
                    const int code = (b[32 * j + m] >> (2 * l)) & 3;
                    s += (code - 1) * (int32_t)y.qs[128 * j + 32 * l + m];
                }
        sumf += d16(b + 64) * y.d * (float)s;
    }
    return sumf;
}

// port of cpu_moe.rs: q4_layout (Q4_0 (T = 2, 18 bytes), Q4_1 (3, 20), Q5_0 (6, 22), Q5_1 (7, 24)): block bytes,
// offset of the 32 high bits (0 = none) and of the 16 bytes of nibbles.
struct Q4Layout {
    size_t bb, oh, oq;
};
constexpr Q4Layout q4_layout(int t) {
    return t == 2 ? Q4Layout{18, 0, 2} : t == 3 ? Q4Layout{20, 0, 4} : t == 6 ? Q4Layout{22, 2, 6} : Q4Layout{24, 4, 8};
}
// port of cpu_moe.rs: q4_min (the m in value = d * q + m)
inline float q4_min(int t, const uint8_t* b) {
    if (t == 3 || t == 7) return d16(b + 2);
    if (t == 2) return -8.0f * d16(b);
    return -16.0f * d16(b);
}

// port of cpu_moe.rs: dot_q4<T>
template <int T>
float dot_q4(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    constexpr Q4Layout L = q4_layout(T);
    float sum = 0.0f;
    const size_t nb = n / L.bb;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + L.bb * i;
        const Q8K& y = xq[i / 8];
        const size_t o = (i % 8) * 32;
        const uint32_t qh = L.oh > 0 ? le32(b + L.oh) : 0u;
        const uint8_t* qs = b + L.oq;
        int32_t s = 0;
        for (size_t j = 0; j < 16; j++) {
            const int32_t l = (int32_t)(qs[j] & 0xF) | (int32_t)(((qh >> j) & 1) * 16);
            const int32_t h = (int32_t)(qs[j] >> 4) | (int32_t)(((qh >> (j + 16)) & 1) * 16);
            s += l * (int32_t)y.qs[o + j] + h * (int32_t)y.qs[o + j + 16];
        }
        sum += y.d * (d16(b) * (float)s + q4_min(T, b) * (float)((int32_t)y.bs[o / 16] + (int32_t)y.bs[o / 16 + 1]));
    }
    return sum;
}

// port of cpu_moe.rs: dot_q8_0 (32 values d * q per block, eight blocks per Q8_K block)
float dot_q8_0(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    float sum = 0.0f;
    const size_t nb = n / 34;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + 34 * i;
        const Q8K& y = xq[i / 8];
        const size_t o = (i % 8) * 32;
        int32_t s = 0;
        for (size_t j = 0; j < 32; j++) s += (int32_t)(int8_t)b[2 + j] * (int32_t)y.qs[o + j];
        sum += d16(b) * y.d * (float)s;
    }
    return sum;
}

// port of cpu_moe.rs: dot_q4k
float dot_q4k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 144, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 144 * bi;
        const Q8K& y = xq[bi];
        uint8_t scv[8], mnv[8];
        for (size_t j = 0; j < 8; j++) scale_min_k4(j, b + 4, scv[j], mnv[j]);
        int8_t a[256];
        for (int i = 0; i < 256; i++) a[i] = (int8_t)((b[16 + 32 * (i / 64) + i % 32] >> (4 * ((i / 32) % 2))) & 0xF);
        const int32_t s = grouped(a, y.qs, [&](int j) { return (int32_t)scv[j / 2]; });
        int32_t m = 0;
        for (int j = 0; j < 8; j++) m += (int32_t)mnv[j] * ((int32_t)y.bs[2 * j] + (int32_t)y.bs[2 * j + 1]);
        sumf += d16(b) * y.d * (float)s - d16(b + 2) * y.d * (float)m;
    }
    return sumf;
}

// port of cpu_moe.rs: dot_q5k (Q4_K plus a fifth bit per value from qh)
float dot_q5k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 176, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 176 * bi;
        const Q8K& y = xq[bi];
        uint8_t scv[8], mnv[8];
        for (size_t j = 0; j < 8; j++) scale_min_k4(j, b + 4, scv[j], mnv[j]);
        int8_t a[256];
        for (int j = 0; j < 4; j++)
            for (int l = 0; l < 32; l++) {
                const uint8_t q = b[48 + 32 * j + l];
                const uint8_t h = (uint8_t)(b[16 + l] >> (2 * j));
                a[64 * j + l] = (int8_t)((q & 0xF) | ((h & 1) << 4));
                a[64 * j + 32 + l] = (int8_t)((q >> 4) | ((h & 2) << 3));
            }
        const int32_t s = grouped(a, y.qs, [&](int j) { return (int32_t)scv[j / 2]; });
        int32_t m = 0;
        for (int j = 0; j < 8; j++) m += (int32_t)mnv[j] * ((int32_t)y.bs[2 * j] + (int32_t)y.bs[2 * j + 1]);
        sumf += d16(b) * y.d * (float)s - d16(b + 2) * y.d * (float)m;
    }
    return sumf;
}

// port of cpu_moe.rs: dot_q6k
float dot_q6k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    float sumf = 0.0f;
    const size_t nb = std::min(n / 210, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 210 * bi;
        const Q8K& y = xq[bi];
        int8_t a[256];
        for (int nn = 0; nn < 2; nn++) {
            const uint8_t* ql = b + 64 * nn;
            const uint8_t* qh = b + 128 + 32 * nn;
            int8_t* aa = a + 128 * nn;
            for (int l = 0; l < 32; l++) {
                aa[l] = (int8_t)(((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32);
                aa[l + 32] = (int8_t)(((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                aa[l + 64] = (int8_t)(((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                aa[l + 96] = (int8_t)(((ql[l + 32] >> 4) | ((qh[l] >> 6) << 4)) - 32);
            }
        }
        int32_t s = 0;
        for (int j = 0; j < 16; j++) {
            int32_t t = 0;
            for (int i = 16 * j; i < 16 * j + 16; i++) t += (int32_t)a[i] * (int32_t)y.qs[i];
            s += (int32_t)(int8_t)b[192 + j] * t;
        }
        sumf += d16(b + 208) * y.d * (float)s;
    }
    return sumf;
}

#if SC_AVX2
// ---------------------------------------------------------------- AVX2 kernels (cpu_moe.rs: mod avx2)
namespace avx2 {

inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    return _mm_cvtss_f32(_mm_add_ss(s, _mm_movehdup_ps(s)));
}

// port of cpu_moe.rs: avx2::quant_block (qs = round(is * b), as the scalar path)
void quant_block(float is, const float* b, int8_t* qs) {
    const __m256 s = _mm256_set1_ps(is);
    auto q = [&](size_t i) {
        return _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(s, _mm256_loadu_ps(b + i)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
    };
    const __m256i perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    for (size_t j = 0; j < 256; j += 32) {
        const __m256i p = _mm256_packs_epi16(_mm256_packs_epi32(q(j), q(j + 8)), _mm256_packs_epi32(q(j + 16), q(j + 24)));
        _mm256_storeu_si256((__m256i*)(qs + j), _mm256_permutevar8x32_epi32(p, perm));
    }
}

inline int64_t sg64(size_t i) {
    int64_t r;
    std::memcpy(&r, signs(i), 8);
    return r;
}
inline __m256i ld256(const uint8_t* p) { return _mm256_loadu_si256((const __m256i*)p); }
inline __m256i q8v(const Q8K& y, size_t i) { return _mm256_loadu_si256((const __m256i*)(y.qs + i)); }
// Pair sums p (16 i16) times lo (first 8 lanes, values 0..16) and hi (last 8), summed into 8 i32.
inline __m256i udot_scale(__m256i p, int lo, int hi) {
    return _mm256_madd_epi16(p, _mm256_set_m128i(_mm_set1_epi16((short)hi), _mm_set1_epi16((short)lo)));
}
inline __m256i udot32(__m256i w, __m256i q, int lo, int hi) { return udot_scale(_mm256_maddubs_epi16(w, q), lo, hi); }
// Crumb k (bits 2k, 2k+1) of every byte.
inline __m256i crumb(__m256i v, int k) {
    return _mm256_and_si256(_mm256_srl_epi16(v, _mm_cvtsi32_si128(2 * k)), _mm256_set1_epi8(3));
}
inline __m256i dot1v(__m256i g, __m256i q) { return _mm256_maddubs_epi16(_mm256_sign_epi8(g, g), _mm256_sign_epi8(q, g)); }

// port of cpu_moe.rs: avx2::dot_iq2xs
float dot_iq2xs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 74, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 74 * bi;
        const Q8K& y = xq[bi];
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            const uint64_t v = le64(b + 2 + 8 * ib);
            const uint16_t q0 = (uint16_t)v, q1 = (uint16_t)(v >> 16), q2 = (uint16_t)(v >> 32), q3 = (uint16_t)(v >> 48);
            auto g = [](uint16_t q) { return (int64_t)IQ2XS_GRID[q & 511]; };
            const __m256i grid = _mm256_set_epi64x(g(q3), g(q2), g(q1), g(q0));
            const __m256i sgn = _mm256_set_epi64x(sg64(q3 >> 9), sg64(q2 >> 9), sg64(q1 >> 9), sg64(q0 >> 9));
            const __m256i q8 = q8v(y, 32 * ib);
            const __m256i dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8, sgn));
            const uint8_t sc = b[66 + ib];
            const __m256i ls = _mm256_set_m128i(_mm_set1_epi16((short)(2 * (sc >> 4) + 1)), _mm_set1_epi16((short)(2 * (sc & 0xF) + 1)));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, ls));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return 0.125f * hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_iq3xxs
float dot_iq3xxs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 98, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 98 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* qs = b + 2;
        const uint8_t* gas = b + 66;
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            const uint64_t v = le64(qs + 8 * ib);
            auto g = [&](int i) { return (int32_t)IQ3XXS_GRID[(uint8_t)(v >> (8 * i))]; };
            const __m256i grid = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
            const uint32_t aux = le32(gas + 4 * ib);
            auto s = [&](uint32_t l) { return sg64((aux >> (7 * l)) & 127); };
            const __m256i sgn = _mm256_set_epi64x(s(3), s(2), s(1), s(0));
            const __m256i q8 = q8v(y, 32 * ib);
            const __m256i dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8, sgn));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16((short)(2 * (int32_t)(aux >> 28) + 1))));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return 0.25f * hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_iq3s
float dot_iq3s(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    // Byte k of a lane picks sign byte k / 8, then bit k % 8 (llama.cpp k_mask1 / k_mask2).
    const __m256i m1 = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    const __m256i m2 = _mm256_set1_epi64x((int64_t)0x8040201008040201ULL);
    const __m256i shl = _mm256_set_epi32(1, 2, 3, 4, 5, 6, 7, 8);
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 110, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 110 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* qs = b + 2;
        const uint8_t* qh = b + 66;
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            const __m256i lo = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(qs + 8 * ib)));
            const __m256i hi = _mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32((int)qh[ib]), shl), _mm256_set1_epi32(256));
            const __m256i grid = _mm256_i32gather_epi32((const int*)IQ3S_GRID, _mm256_or_si256(lo, hi), 4);
            const int sb = (int)le32(b + 74 + 4 * ib);
            const __m256i s = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2), m2);
            const __m256i q8 = q8v(y, 32 * ib);
            const __m256i dot = _mm256_maddubs_epi16(grid, _mm256_sub_epi8(_mm256_xor_si256(s, q8), s));
            const int16_t ls = (int16_t)(2 * ((b[106 + ib / 2] >> (4 * (ib % 2))) & 0xF) + 1);
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16(ls)));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_iq4xs
float dot_iq4xs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    const __m128i values = _mm_loadu_si128((const __m128i*)KVALUES_IQ4NL);
    const __m128i m4 = _mm_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 136, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 136 * bi;
        const Q8K& y = xq[bi];
        const uint16_t sh = le16(b + 2);
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            const __m128i bits = _mm_loadu_si128((const __m128i*)(b + 8 + 16 * ib));
            const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                                               _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)));
            const __m256i q8 = q8v(y, 32 * ib);
            const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8, w));
            const int16_t ls = (int16_t)(((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | ((((sh >> (2 * ib)) & 3)) << 4));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p16, _mm256_set1_epi16((short)(ls - 32))));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_iq4nl (two 32-value blocks per step so the two lookups and FMAs overlap)
float dot_iq4nl(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    const __m128i values = _mm_loadu_si128((const __m128i*)KVALUES_IQ4NL);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    auto blk = [&](const uint8_t* b, const Q8K& y, size_t o, __m256& s_out, __m256& d_out) {
        const __m128i bits = _mm_loadu_si128((const __m128i*)(b + 2));
        const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                                           _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8v(y, o), w));
        s_out = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
        d_out = _mm256_set1_ps(d16(b) * y.d);
    };
    const size_t n_blocks = n / 18;
    const size_t n_pairs = n / 36;
    for (size_t i = 0; i < n_pairs; i++) {
        const Q8K& y = xq[(2 * i) / 8];
        const size_t o = (2 * i) % 8 * 32;
        __m256 s0, d0, s1, d1;
        blk(row + 36 * i, y, o, s0, d0);
        blk(row + 36 * i + 18, y, o + 32, s1, d1);
        acc0 = _mm256_fmadd_ps(d0, s0, acc0);
        acc1 = _mm256_fmadd_ps(d1, s1, acc1);
    }
    if (n_blocks % 2 == 1) {
        const size_t i = n_blocks - 1;
        __m256 s, d;
        blk(row + 18 * i, xq[i / 8], i % 8 * 32, s, d);
        acc0 = _mm256_fmadd_ps(d, s, acc0);
    }
    return hsum(_mm256_add_ps(acc0, acc1));
}

// StrataFreet addition: avx2::dot_mxfp4, the structure of avx2::dot_iq4nl with 17-byte blocks (codes at b + 1, the
// scale e8m0_half(b[0]) * y.d). The 16-byte code load at b + 1 stays inside the 17-byte block.
float dot_mxfp4(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    const __m128i values = _mm_loadu_si128((const __m128i*)kKvaluesMxfp4);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
    auto blk = [&](const uint8_t* b, const Q8K& y, size_t o, __m256& s_out, __m256& d_out) {
        const __m128i bits = _mm_loadu_si128((const __m128i*)(b + 1));
        const __m256i w = _mm256_set_m128i(_mm_shuffle_epi8(values, _mm_and_si128(_mm_srli_epi16(bits, 4), m4)),
                                           _mm_shuffle_epi8(values, _mm_and_si128(bits, m4)));
        const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8v(y, o), w));
        s_out = _mm256_cvtepi32_ps(_mm256_madd_epi16(p16, ones));
        d_out = _mm256_set1_ps(e8m0_half(b[0]) * y.d);
    };
    const size_t n_blocks = n / 17;
    const size_t n_pairs = n / 34;
    for (size_t i = 0; i < n_pairs; i++) {
        const Q8K& y = xq[(2 * i) / 8];
        const size_t o = (2 * i) % 8 * 32;
        __m256 s0, d0, s1, d1;
        blk(row + 34 * i, y, o, s0, d0);
        blk(row + 34 * i + 17, y, o + 32, s1, d1);
        acc0 = _mm256_fmadd_ps(d0, s0, acc0);
        acc1 = _mm256_fmadd_ps(d1, s1, acc1);
    }
    if (n_blocks % 2 == 1) {
        const size_t i = n_blocks - 1;
        __m256 s, d;
        blk(row + 17 * i, xq[i / 8], i % 8 * 32, s, d);
        acc0 = _mm256_fmadd_ps(d, s, acc0);
    }
    return hsum(_mm256_add_ps(acc0, acc1));
}

// port of cpu_moe.rs: avx2::dot_iq2xxs
float dot_iq2xxs(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 66, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 66 * bi;
        const Q8K& y = xq[bi];
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            auto g = [&](int l) { return (int64_t)IQ2XXS_GRID[b[2 + 8 * ib + l]]; };
            const uint32_t aux = le32(b + 6 + 8 * ib);
            auto s = [&](uint32_t l) { return sg64((aux >> (7 * l)) & 127); };
            const __m256i grid = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
            const __m256i dot = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8v(y, 32 * ib), _mm256_set_epi64x(s(3), s(2), s(1), s(0))));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_set1_epi16((short)(2 * (int32_t)(aux >> 28) + 1))));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return 0.125f * hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_iq2s
float dot_iq2s(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    // As in dot_iq3s: byte k of a lane picks sign byte k / 8, then bit k % 8.
    const __m256i m1 = _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0);
    const __m256i m2 = _mm256_set1_epi64x((int64_t)0x8040201008040201ULL);
    // Grid index high bits: word p of the 32 indices takes qh[p / 4] << (8 - 2 * (p % 4)) & 0x300.
    const __m256i hq0 = _mm256_set_epi64x((int64_t)0x8003800380038003ULL, (int64_t)0x8002800280028002ULL,
                                          (int64_t)0x8001800180018001ULL, (int64_t)0x8000800080008000ULL);
    const __m256i hq1 = _mm256_add_epi16(hq0, _mm256_set1_epi16(4));
    const __m256i hmul = _mm256_set1_epi64x((int64_t)0x0004001000400100ULL);
    const __m256i h3 = _mm256_set1_epi16(0x300);
    // Scale words: the low lane takes byte 2 ib of the 16 block scales, the high lane byte 2 ib + 1.
    const __m256i sm0 = _mm256_set_epi64x((int64_t)0x8001800180018001ULL, (int64_t)0x8001800180018001ULL,
                                          (int64_t)0x8000800080008000ULL, (int64_t)0x8000800080008000ULL);
    const __m256i sstep = _mm256_set1_epi16(2);
    const __m128i n15 = _mm_set1_epi8(15);
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 82, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 82 * bi;
        const Q8K& y = xq[bi];
        const __m256i qh = _mm256_broadcastsi128_si256(_mm_loadl_epi64((const __m128i*)(b + 66)));
        uint16_t idx[32];
        const __m256i hqs[2] = {hq0, hq1};
        for (int h = 0; h < 2; h++) {
            const __m256i lo = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i*)(b + 2 + 16 * h)));
            const __m256i hi = _mm256_and_si256(_mm256_mullo_epi16(_mm256_shuffle_epi8(qh, hqs[h]), hmul), h3);
            _mm256_storeu_si256((__m256i*)(idx + 16 * h), _mm256_or_si256(lo, hi));
        }
        __m128i sc128 = _mm_loadl_epi64((const __m128i*)(b + 74));
        sc128 = _mm_unpacklo_epi8(_mm_and_si128(sc128, n15), _mm_and_si128(_mm_srli_epi16(sc128, 4), n15));
        const __m256i sc = _mm256_broadcastsi128_si256(_mm_add_epi8(_mm_add_epi8(sc128, sc128), _mm_set1_epi8(1)));
        __m256i sm = sm0;
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            auto g = [&](int l) { return (int64_t)IQ2S_GRID[idx[4 * ib + l] & 1023]; };
            const __m256i grid = _mm256_set_epi64x(g(3), g(2), g(1), g(0));
            const int sb = (int)le32(b + 34 + 4 * ib);
            const __m256i s = _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(_mm256_set1_epi32(sb), m1), m2), m2);
            const __m256i dot = _mm256_maddubs_epi16(grid, _mm256_sub_epi8(_mm256_xor_si256(s, q8v(y, 32 * ib)), s));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(dot, _mm256_shuffle_epi8(sc, sm)));
            sm = _mm256_add_epi16(sm, sstep);
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return 0.125f * hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_iq1s
float dot_iq1s(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    float sumf = 0.0f;
    const size_t nb = std::min(n / 50, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 50 * bi;
        const Q8K& y = xq[bi];
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            const size_t qh = le16(b + 34 + 2 * ib);
            auto g = [&](int l) { return (int64_t)IQ1S_GRID[(size_t)b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)]; };
            const int32_t ls = 2 * (int32_t)((qh >> 12) & 7) + 1;
            const __m256i p = dot1v(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), q8v(y, 32 * ib));
            sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, _mm256_set1_epi16((short)ls)));
        }
        // Delta term: sum of +-scale (sign in bit 15 of qh) times each 32 values' q8 sum.
        const __m128i qh = _mm_loadu_si128((const __m128i*)(b + 34));
        __m128i ls = _mm_or_si128(_mm_slli_epi16(_mm_and_si128(_mm_srli_epi16(qh, 12), _mm_set1_epi16(7)), 1), _mm_set1_epi16(1));
        ls = _mm_sign_epi16(ls, _mm_or_si128(qh, _mm_set1_epi16(1)));
        auto bsv = [&](int i) { return _mm_loadu_si128((const __m128i*)(y.bs + 8 * i)); };
        __m128i s = _mm_add_epi32(_mm_madd_epi16(_mm_unpacklo_epi16(ls, ls), bsv(0)), _mm_madd_epi16(_mm_unpackhi_epi16(ls, ls), bsv(1)));
        s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4E));           // 0b01_00_11_10
        const int32_t s1 = _mm_cvtsi128_si32(_mm_add_epi32(s, _mm_shuffle_epi32(s, 0xB1)));   // 0b10_11_00_01
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
        sumf += d16(b) * y.d * 0.125f * (float)s1;
    }
    return hsum(acc) + sumf;
}

// IQ1_M weights plus 9 (8 * grid + 8, plus 2 when the delta is +1), indexed by the 8 grid bits, then 3 high grid
// bits and the delta sign bit (a qh nibble). port of cpu_moe.rs: IQ1M_W (a LazyLock in Rust).
const uint64_t* iq1m_w() {
    static const uint64_t* table = [] {
        uint64_t* t = new uint64_t[4096];
        for (size_t i = 0; i < 4096; i++) {
            const int d = (i & 2048) ? 8 : 10;
            uint8_t g[8];
            grid_bytes8(IQ1S_GRID[i & 2047], g);
            uint8_t out[8];
            for (int k = 0; k < 8; k++) out[k] = (uint8_t)(8 * (int)(int8_t)g[k] + d);
            std::memcpy(&t[i], out, 8);
        }
        return (const uint64_t*)t;
    }();
    return table;
}

// port of cpu_moe.rs: avx2::dot_iq1m
float dot_iq1m(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    const uint64_t* t = iq1m_w();
    const __m256i nine = _mm256_set1_epi8(9);
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 56, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 56 * bi;
        const Q8K& y = xq[bi];
        // Weights 8 * grid + delta (-9..=9), so one 0.125 at the end gives grid + delta / 8. The table holds them
        // plus 9 (unsigned for maddubs), taken off the pair sums again with 9 * q8.
        const uint64_t sc = le64(b + 48);
        int16_t lsv[16];
        for (int i = 0; i < 16; i++) lsv[i] = (int16_t)(2 * (int32_t)((sc >> (16 * (i / 4) + 3 * (i % 4))) & 7) + 1);
        __m256i sumi = _mm256_setzero_si256();
        for (int ib = 0; ib < 8; ib++) {
            const size_t qh = le16(b + 32 + 2 * ib);
            auto g = [&](int l) { return (int64_t)t[(size_t)b[4 * ib + l] | (((qh >> (4 * l)) & 15) << 8)]; };
            const __m256i q = q8v(y, 32 * ib);
            const __m256i p = _mm256_sub_epi16(_mm256_maddubs_epi16(_mm256_set_epi64x(g(3), g(2), g(1), g(0)), q),
                                               _mm256_maddubs_epi16(nine, q));
            sumi = _mm256_add_epi32(sumi, udot_scale(p, lsv[2 * ib], lsv[2 * ib + 1]));
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(iq1m_d(b) * y.d), _mm256_cvtepi32_ps(sumi), acc);
    }
    return 0.125f * hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_q2k
float dot_q2k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 84, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 84 * bi;
        const Q8K& y = xq[bi];
        const uint8_t* sc = b;
        __m256i sumi = _mm256_setzero_si256();
        for (int nn = 0; nn < 2; nn++) {
            const __m256i bits = ld256(b + 16 + 32 * nn);
            for (int k = 0; k < 4; k++) {
                const int g = 8 * nn + 2 * k;
                sumi = _mm256_add_epi32(sumi, udot32(crumb(bits, k), q8v(y, 32 * (4 * nn + k)), sc[g] & 0xF, sc[g + 1] & 0xF));
            }
        }
        const __m256i mins = _mm256_cvtepu8_epi16(_mm_and_si128(_mm_srli_epi16(_mm_loadu_si128((const __m128i*)sc), 4), _mm_set1_epi8(0xF)));
        const __m256i m = _mm256_madd_epi16(mins, _mm256_loadu_si256((const __m256i*)y.bs));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b + 80)), _mm256_cvtepi32_ps(sumi), acc);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(-y.d * d16(b + 82)), _mm256_cvtepi32_ps(m), acc);
    }
    return hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_q3k
float dot_q3k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = std::min(n / 110, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 110 * bi;
        const Q8K& y = xq[bi];
        const __m256i hm = ld256(b);
        int8_t sc[16];
        q3k_scales(b, sc);
        __m256i sumi = _mm256_setzero_si256();
        for (int nn = 0; nn < 2; nn++) {
            const __m256i bits = ld256(b + 32 + 32 * nn);
            for (int k = 0; k < 4; k++) {
                const int kk = 4 * nn + k;
                // Weight = crumb - 4 where the high bit is clear; signs moved onto q8 for maddubs.
                const __m256i mask = _mm256_set1_epi8((char)(1 << kk));
                const __m256i low = _mm256_cmpeq_epi8(_mm256_and_si256(hm, mask), _mm256_setzero_si256());
                const __m256i w = _mm256_sub_epi8(crumb(bits, k), _mm256_and_si256(low, _mm256_set1_epi8(4)));
                const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8v(y, 32 * kk), w));
                const __m256i s = _mm256_set_m128i(_mm_set1_epi16((short)(sc[2 * kk + 1] - 32)), _mm_set1_epi16((short)(sc[2 * kk] - 32)));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(p, s));
            }
        }
        acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b + 108)), _mm256_cvtepi32_ps(sumi), acc);
    }
    return hsum(acc);
}

// port of cpu_moe.rs: q2_0_crumbs (the 64 crumbs of a Q2_0 block in element order, as two vectors of 32 bytes;
// this is the AVX2 form, the AVX-512 VBMI form of the Rust code computes the same bytes)
inline void q2_0_crumbs(const uint8_t* b, __m256i& w0, __m256i& w1) {
    const __m128i v = _mm_loadu_si128((const __m128i*)(b + 2));
    auto c = [&](int k) { return _mm_and_si128(_mm_srl_epi16(v, _mm_cvtsi32_si128(2 * k)), _mm_set1_epi8(3)); };
    const __m128i c0 = c(0), c1 = c(1), c2 = c(2), c3 = c(3);
    const __m128i lo01 = _mm_unpacklo_epi8(c0, c1), lo23 = _mm_unpacklo_epi8(c2, c3);
    const __m128i hi01 = _mm_unpackhi_epi8(c0, c1), hi23 = _mm_unpackhi_epi8(c2, c3);
    w0 = _mm256_set_m128i(_mm_unpackhi_epi16(lo01, lo23), _mm_unpacklo_epi16(lo01, lo23));
    w1 = _mm256_set_m128i(_mm_unpackhi_epi16(hi01, hi23), _mm_unpacklo_epi16(hi01, hi23));
}

// port of cpu_moe.rs: avx2::dot_q2_0 (row.chunks(72) zipped with xq: four blocks per Q8_K block)
float dot_q2_0(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    __m256 acc = _mm256_setzero_ps();
    float sumf = 0.0f;
    const __m256i ones = _mm256_set1_epi16(1);
    size_t off = 0;
    for (size_t c = 0; c < nx && off < n; c++, off += 72) {
        const size_t len = std::min<size_t>(72, n - off);
        const uint8_t* sb = row + off;
        const Q8K& y = xq[c];
        // The sums of y over each 64 values, in lanes 0, 1, 4 and 5.
        const __m256i bs = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i*)y.bs), ones);
        float m[8];
        _mm256_storeu_ps(m, _mm256_cvtepi32_ps(_mm256_hadd_epi32(bs, bs)));
        for (size_t j = 0; j < len / 18; j++) {
            const uint8_t* b = sb + 18 * j;
            __m256i w0, w1;
            q2_0_crumbs(b, w0, w1);
            const __m256i p = _mm256_add_epi16(_mm256_maddubs_epi16(w0, q8v(y, 64 * j)), _mm256_maddubs_epi16(w1, q8v(y, 64 * j + 32)));
            const float d = d16(b) * y.d;
            acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), acc);
            sumf -= d * m[j + j / 2 * 2];
        }
    }
    return hsum(acc) + sumf;
}

// port of cpu_moe.rs: avx2::dot_q4<T> (Q4_0, Q4_1, Q5_0, Q5_1)
template <int T>
float dot_q4(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    constexpr Q4Layout L = q4_layout(T);
    const __m128i m4 = _mm_set1_epi8(0xF);
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    float sumf = 0.0f;
    const size_t nb = n / L.bb;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + L.bb * i;
        const Q8K& y = xq[i / 8];
        const size_t o = (i % 8) * 32;
        const __m128i q = _mm_loadu_si128((const __m128i*)(b + L.oq));
        __m256i w = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m4), _mm_and_si128(q, m4));
        if (L.oh > 0) {  // bit j of qh as 16 in byte j (llama.cpp bytes_from_bits_32)
            const int qh = (int)le32(b + L.oh);
            const __m256i sh = _mm256_shuffle_epi8(_mm256_set1_epi32(qh), _mm256_set_epi64x(0x0303030303030303LL, 0x0202020202020202LL, 0x0101010101010101LL, 0));
            const __m256i set = _mm256_cmpeq_epi8(_mm256_or_si256(sh, _mm256_set1_epi64x((int64_t)0x7fbfdfeff7fbfdfeULL)), _mm256_set1_epi64x(-1));
            w = _mm256_or_si256(w, _mm256_and_si256(set, _mm256_set1_epi8(16)));
        }
        const __m256i s = _mm256_madd_epi16(_mm256_maddubs_epi16(w, q8v(y, o)), ones);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(s), acc);
        sumf += q4_min(T, b) * y.d * (float)((int32_t)y.bs[o / 16] + (int32_t)y.bs[o / 16 + 1]);
    }
    return hsum(acc) + sumf;
}

// port of cpu_moe.rs: avx2::dot_q4k
float dot_q4k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    float sumf = 0.0f;
    const size_t nb = std::min(n / 144, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 144 * bi;
        const Q8K& y = xq[bi];
        uint8_t scv[8], mnv[8];
        for (size_t j = 0; j < 8; j++) scale_min_k4(j, b + 4, scv[j], mnv[j]);
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {
            const __m256i bits = ld256(b + 16 + 32 * j);
            const int s0 = scv[2 * j], s1 = scv[2 * j + 1];
            sumi = _mm256_add_epi32(sumi, udot32(_mm256_and_si256(bits, m4), q8v(y, 64 * j), s0, s0));
            sumi = _mm256_add_epi32(sumi, udot32(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), q8v(y, 64 * j + 32), s1, s1));
        }
        int32_t m = 0;
        for (int j = 0; j < 8; j++) m += (int32_t)mnv[j] * ((int32_t)y.bs[2 * j] + (int32_t)y.bs[2 * j + 1]);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b)), _mm256_cvtepi32_ps(sumi), acc);
        sumf -= y.d * d16(b + 2) * (float)m;
    }
    return hsum(acc) + sumf;
}

// port of cpu_moe.rs: avx2::dot_q8_0 (|w| times q8 with w's sign, as maddubs needs an unsigned first operand)
float dot_q8_0(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    (void)nx;
    const __m256i ones = _mm256_set1_epi16(1);
    __m256 acc = _mm256_setzero_ps();
    const size_t nb = n / 34;
    for (size_t i = 0; i < nb; i++) {
        const uint8_t* b = row + 34 * i;
        const Q8K& y = xq[i / 8];
        const size_t o = (i % 8) * 32;
        const __m256i w = _mm256_loadu_si256((const __m256i*)(b + 2));
        const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(q8v(y, o), w));
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d16(b) * y.d), _mm256_cvtepi32_ps(_mm256_madd_epi16(p, ones)), acc);
    }
    return hsum(acc);
}

// port of cpu_moe.rs: avx2::dot_q5k
float dot_q5k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i one = _mm256_set1_epi8(1);
    // Bit k of every qh byte as 16.
    auto hb = [&](__m256i qh, int k) {
        return _mm256_slli_epi16(_mm256_and_si256(_mm256_srl_epi16(qh, _mm_cvtsi32_si128(k)), one), 4);
    };
    __m256 acc = _mm256_setzero_ps();
    float sumf = 0.0f;
    const size_t nb = std::min(n / 176, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 176 * bi;
        const Q8K& y = xq[bi];
        uint8_t scv[8], mnv[8];
        for (size_t j = 0; j < 8; j++) scale_min_k4(j, b + 4, scv[j], mnv[j]);
        const __m256i qh = ld256(b + 16);
        __m256i sumi = _mm256_setzero_si256();
        for (int j = 0; j < 4; j++) {
            const __m256i bits = ld256(b + 48 + 32 * j);
            const int s0 = scv[2 * j], s1 = scv[2 * j + 1];
            const __m256i w0 = _mm256_or_si256(_mm256_and_si256(bits, m4), hb(qh, 2 * j));
            const __m256i w1 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(bits, 4), m4), hb(qh, 2 * j + 1));
            sumi = _mm256_add_epi32(sumi, udot32(w0, q8v(y, 64 * j), s0, s0));
            sumi = _mm256_add_epi32(sumi, udot32(w1, q8v(y, 64 * j + 32), s1, s1));
        }
        int32_t m = 0;
        for (int j = 0; j < 8; j++) m += (int32_t)mnv[j] * ((int32_t)y.bs[2 * j] + (int32_t)y.bs[2 * j + 1]);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(y.d * d16(b)), _mm256_cvtepi32_ps(sumi), acc);
        sumf -= y.d * d16(b + 2) * (float)m;
    }
    return hsum(acc) + sumf;
}

// port of cpu_moe.rs: avx2::dot_q6k (unsigned 6-bit codes; the -32 offset comes off through the Q8_K group sums)
float dot_q6k(const uint8_t* row, size_t n, const Q8K* xq, size_t nx) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    __m256 acc = _mm256_setzero_ps();
    float sumf = 0.0f;
    const size_t nb = std::min(n / 210, nx);
    for (size_t bi = 0; bi < nb; bi++) {
        const uint8_t* b = row + 210 * bi;
        const Q8K& y = xq[bi];
        auto sc = [&](size_t g) { return (int32_t)(int8_t)b[192 + g]; };
        __m256i sumi = _mm256_setzero_si256();
        for (int nn = 0; nn < 2; nn++) {
            const __m256i la = ld256(b + 64 * nn), lb = ld256(b + 64 * nn + 32), qh = ld256(b + 128 + 32 * nn);
            auto hi = [&](int k) { return _mm256_slli_epi16(crumb(qh, k), 4); };
            const __m256i w[4] = {
                _mm256_or_si256(_mm256_and_si256(la, m4), hi(0)),
                _mm256_or_si256(_mm256_and_si256(lb, m4), hi(1)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(la, 4), m4), hi(2)),
                _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(lb, 4), m4), hi(3)),
            };
            for (int k = 0; k < 4; k++) {
                const size_t g = 8 * nn + 2 * k;
                sumi = _mm256_add_epi32(sumi, udot32(w[k], q8v(y, 128 * nn + 32 * k), (int)sc(g), (int)sc(g + 1)));
            }
        }
        int32_t m = 0;
        for (size_t g = 0; g < 16; g++) m += (int32_t)sc(g) * (int32_t)y.bs[g];
        const float d = y.d * d16(b + 208);
        acc = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), acc);
        sumf -= 32.0f * d * (float)m;
    }
    return hsum(acc) + sumf;
}

}  // namespace avx2
#endif  // SC_AVX2

// ---------------------------------------------------------------- kernel selection (cpu_moe.rs: int_dot)

// The scalar kernel of a type, or nullptr for the types without an integer kernel (F32, F16, BF16).
DotFn scalar_dot(Ty ty) {
    switch (ty) {
        case Ty::IQ2XS: return dot_iq2xs;
        case Ty::IQ3XXS: return dot_iq3xxs;
        case Ty::IQ4XS: return dot_iq4xs;
        case Ty::IQ4NL: return dot_iq4nl;
        case Ty::MXFP4: return dot_mxfp4;
        case Ty::IQ3S: return dot_iq3s;
        case Ty::Q6K: return dot_q6k;
        case Ty::Q5K: return dot_q5k;
        case Ty::Q8_0: return dot_q8_0;
        case Ty::IQ2XXS: return dot_iq2xxs;
        case Ty::IQ2S: return dot_iq2s;
        case Ty::IQ1S: return dot_iq1s;
        case Ty::IQ1M: return dot_iq1m;
        case Ty::Q2K: return dot_q2k;
        case Ty::Q3K: return dot_q3k;
        case Ty::Q4K: return dot_q4k;
        case Ty::Q2_0: return dot_q2_0;
        case Ty::TQ1_0: return dot_tq1_0;
        case Ty::TQ2_0: return dot_tq2_0;
        case Ty::Q4_0: return dot_q4<2>;
        case Ty::Q4_1: return dot_q4<3>;
        case Ty::Q5_0: return dot_q4<6>;
        case Ty::Q5_1: return dot_q4<7>;
        default: return nullptr;
    }
}

#if SC_AVX2
// The AVX2 kernel of a type; the types without one keep their scalar kernel (as the Rust match does).
DotFn avx2_dot(Ty ty) {
    switch (ty) {
        case Ty::IQ2XS: return avx2::dot_iq2xs;
        case Ty::IQ3XXS: return avx2::dot_iq3xxs;
        case Ty::IQ4XS: return avx2::dot_iq4xs;
        case Ty::IQ4NL: return avx2::dot_iq4nl;
        case Ty::MXFP4: return avx2::dot_mxfp4;
        case Ty::IQ3S: return avx2::dot_iq3s;
        case Ty::Q6K: return avx2::dot_q6k;
        case Ty::Q5K: return avx2::dot_q5k;
        case Ty::Q8_0: return avx2::dot_q8_0;
        case Ty::IQ2XXS: return avx2::dot_iq2xxs;
        case Ty::IQ2S: return avx2::dot_iq2s;
        case Ty::IQ1S: return avx2::dot_iq1s;
        case Ty::IQ1M: return avx2::dot_iq1m;
        case Ty::Q2K: return avx2::dot_q2k;
        case Ty::Q3K: return avx2::dot_q3k;
        case Ty::Q4K: return avx2::dot_q4k;
        case Ty::Q2_0: return avx2::dot_q2_0;
        case Ty::Q4_0: return avx2::dot_q4<2>;
        case Ty::Q4_1: return avx2::dot_q4<3>;
        case Ty::Q5_0: return avx2::dot_q4<6>;
        case Ty::Q5_1: return avx2::dot_q4<7>;
        default: return scalar_dot(ty);
    }
}
#endif

// port of cpu_moe.rs: int_dot. `scalar` is the STRATA_SCALAR switch (the Rust code reads it at each call).
DotFn int_dot(Ty ty, bool scalar) {
#if SC_AVX2
    if (!scalar) return avx2_dot(ty);
#else
    (void)scalar;
#endif
    return scalar_dot(ty);
}

bool scalar_env() { return std::getenv("STRATA_SCALAR") != nullptr; }

// ---------------------------------------------------------------- activations (cpu_moe.rs: quantize_q8k)

#if SC_AVX2
void quant_block_dispatch(float is, const float* b, int8_t* qs) { avx2::quant_block(is, b, qs); }
#endif

// ---------------------------------------------------------------- matrix helpers (cpu_moe.rs: batch_rows, rows_multi)

// port of cpu_moe.rs: batch_rows. Rows of a batch of matrices: out[i] = mats[i / rows] row (i % rows) . x of that
// matrix, where matrix j uses xs[j * xs.len() / mats.len()] (one x for all, or one each). A shared x is quantized
// once. Tasks are chunks of 16 rows, or of 64 for Q2_0 on AVX2 when the rows fill the pool (the Rust rule: "Q2_0
// experts benefit from fewer Rayon tasks on AVX2"; a chunk is still computed by one task, so results are unchanged).
void batch_rows(const std::vector<Tensor>& mats, const std::vector<const float*>& xs, size_t x_len, float* out,
                size_t out_len, ThreadPool& pool) {
    const size_t rows = out_len / mats.size();
    const DotFn dot = int_dot(mats[0].ty, scalar_env());
    std::vector<std::vector<Q8K>> xq;
    if (dot)
        for (const float* x : xs) xq.push_back(quantize_q8k(x, x_len));
    const size_t batch = (mats[0].ty == Ty::Q2_0 && SC_AVX2 && out_len / 64 >= pool.size()) ? 64 : 16;
    const size_t n_tasks = (out_len + batch - 1) / batch;
    pool.for_each(n_tasks, [&](size_t c) {
        std::vector<float> buf;
        if (!dot) buf.resize(x_len);
        const size_t lo = c * batch, hi = std::min(out_len, lo + batch);
        for (size_t i = lo; i < hi; i++) {
            const size_t mi = i / rows, r = i % rows, xi = mi * xs.size() / mats.size();
            const Tensor& t = mats[mi];
            const size_t rb = t.row_bytes();
            if (dot) {
                out[i] = dot(t.data + r * rb, rb, xq[xi].data(), xq[xi].size());
            } else {
                t.row(r, buf.data());
                float s = 0.0f;
                for (size_t k = 0; k < x_len; k++) s += buf[k] * xs[xi][k];
                out[i] = s;
            }
        }
    });
}

// port of cpu_moe.rs: matvec. y = t . x for every row, using the integer kernel when the type has one.
void matvec(const Tensor& t, const float* x, size_t x_len, float* y, ThreadPool& pool) {
    batch_rows({t}, {x}, x_len, y, t.rows, pool);
}

// port of cpu_moe.rs: rows_multi. Every weight row is read once and dotted with all inputs. Returns nt vectors of
// `rows` values.
std::vector<std::vector<float>> rows_multi(const Tensor& t, const std::vector<const float*>& xs, size_t x_len, ThreadPool& pool) {
    const size_t rows = t.rows, nt = xs.size();
    const DotFn dot = int_dot(t.ty, scalar_env());
    std::vector<std::vector<Q8K>> xq;
    if (dot)
        for (const float* x : xs) xq.push_back(quantize_q8k(x, x_len));
    const size_t rb = t.row_bytes();
    std::vector<float> tmp(rows * nt, 0.0f);
    constexpr size_t batch = 16;
    const size_t n_tasks = (rows + batch - 1) / batch;
    pool.for_each(n_tasks, [&](size_t ci) {
        std::vector<float> buf;
        if (!dot) buf.resize(x_len);
        const size_t hi = std::min(rows, ci * batch + batch);
        for (size_t row = ci * batch; row < hi; row++) {
            const uint8_t* data = t.data + row * rb;
            for (size_t j = 0; j < nt; j++) {
                if (dot) {
                    tmp[row * nt + j] = dot(data, rb, xq[j].data(), xq[j].size());
                } else {
                    t.row(row, buf.data());
                    float s = 0.0f;
                    for (size_t k = 0; k < x_len; k++) s += buf[k] * xs[j][k];
                    tmp[row * nt + j] = s;
                }
            }
        }
    });
    std::vector<std::vector<float>> out(nt, std::vector<float>(rows));
    for (size_t j = 0; j < nt; j++)
        for (size_t r = 0; r < rows; r++) out[j][r] = tmp[r * nt + j];
    return out;
}

// ---------------------------------------------------------------- self-test (cpu_moe.rs: int_dots_match_dequant)

std::string type_name(Ty ty) { return ty_name(ty); }

}  // namespace

// ---------------------------------------------------------------- thread pool (rayon's global pool)

ThreadPool::ThreadPool(size_t n_threads) {
    const size_t n = n_threads == 0 ? 1 : n_threads;
    th_.reserve(n);
    for (size_t i = 0; i < n; i++) th_.emplace_back([this] { worker(); });
}

ThreadPool::~ThreadPool() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        stop_.store(true, std::memory_order_release);
    }
    work_cv_.notify_all();
    for (std::thread& t : th_) t.join();
}

// Pool threads spin on the generation counter between passes (a token runs a pass per MoE layer and per expert matrix,
// tens of microseconds apart; a sleeping thread takes far longer to wake than a pass takes), and sleep on the
// condition variable only after kSpins empty polls. Rayon's pool spins the same way.
void ThreadPool::worker() {
    constexpr unsigned kSpins = 200000;
    uint64_t seen = 0;
    for (;;) {
        for (unsigned spins = 0;; spins++) {
            if (gen_.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_acquire)) break;
            if (spins < kSpins) {
                _mm_pause();
                continue;
            }
            std::unique_lock<std::mutex> lk(mu_);
            sleepers_.fetch_add(1, std::memory_order_seq_cst);
            work_cv_.wait(lk, [&] { return stop_.load(std::memory_order_acquire) || gen_.load(std::memory_order_acquire) != seen; });
            sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            break;
        }
        if (stop_.load(std::memory_order_acquire)) return;
        seen = gen_.load(std::memory_order_acquire);
        // The job fields were written before the generation was published and stay unchanged until every thread
        // has acknowledged it (for_each waits for that).
        const std::function<void(size_t)>* f = job_;
        const size_t n = n_;
        if (trace_) {   // STRATA_TRACE_WORKER: start latency, busy time and items of this thread for this job
            const int64_t t0 = now_ns();
            s_lat_.fetch_add(static_cast<uint64_t>(t0 - pub_ns_.load(std::memory_order_acquire)), std::memory_order_relaxed);
            uint64_t items = 0;
            for (;;) {
                const size_t i = next_.fetch_add(1, std::memory_order_acq_rel);
                if (i >= n) break;
                (*f)(i);
                items++;
                finished_.fetch_add(1, std::memory_order_acq_rel);
            }
            s_busy_.fetch_add(static_cast<uint64_t>(now_ns() - t0), std::memory_order_relaxed);
            s_items_.fetch_add(items, std::memory_order_relaxed);
            s_jobs_.fetch_add(1, std::memory_order_relaxed);
            acked_.fetch_add(1, std::memory_order_acq_rel);
            continue;
        }
        for (;;) {
            const size_t i = next_.fetch_add(1, std::memory_order_acq_rel);
            if (i >= n) break;
            (*f)(i);
            finished_.fetch_add(1, std::memory_order_acq_rel);
        }
        acked_.fetch_add(1, std::memory_order_acq_rel);
    }
}

// port of rayon's for_each: the calling thread takes part (the Rust worker loop runs its passes on the pool with
// pool.install, so its thread works alongside the pool's), then waits until every item has run and every pool thread
// has left the job.
void ThreadPool::for_each(size_t n, const std::function<void(size_t)>& f) {
    if (n == 0) return;
    job_ = &f;
    n_ = n;
    next_.store(0, std::memory_order_relaxed);
    finished_.store(0, std::memory_order_relaxed);
    acked_.store(0, std::memory_order_relaxed);
    const int64_t t_start = trace_ ? now_ns() : 0;
    if (trace_) pub_ns_.store(t_start, std::memory_order_release);
    {
        std::lock_guard<std::mutex> lk(mu_);
        gen_.fetch_add(1, std::memory_order_acq_rel);
    }
    if (sleepers_.load(std::memory_order_seq_cst) > 0) work_cv_.notify_all();
    const int64_t t_mine = trace_ ? now_ns() : 0;
    uint64_t items = 0;
    for (;;) {
        const size_t i = next_.fetch_add(1, std::memory_order_acq_rel);
        if (i >= n) break;
        f(i);
        items++;
        finished_.fetch_add(1, std::memory_order_acq_rel);
    }
    const int64_t t_wait = trace_ ? now_ns() : 0;
    while (finished_.load(std::memory_order_acquire) < n || acked_.load(std::memory_order_acquire) < th_.size()) {
        _mm_pause();
    }
    if (trace_) {
        const int64_t t_end = now_ns();
        c_total_.fetch_add(static_cast<uint64_t>(t_end - t_start), std::memory_order_relaxed);
        c_wait_.fetch_add(static_cast<uint64_t>(t_end - t_wait), std::memory_order_relaxed);
        c_busy_.fetch_add(static_cast<uint64_t>(t_wait - t_mine), std::memory_order_relaxed);
        c_items_.fetch_add(items, std::memory_order_relaxed);
        c_calls_.fetch_add(1, std::memory_order_relaxed);
    }
    job_ = nullptr;
}

ThreadPool::Stats ThreadPool::stats() const {
    Stats st;
    st.jobs = s_jobs_.load();
    st.start_lat_ns = s_lat_.load();
    st.busy_ns = s_busy_.load();
    st.items = s_items_.load();
    st.caller_total_ns = c_total_.load();
    st.caller_wait_ns = c_wait_.load();
    st.caller_busy_ns = c_busy_.load();
    st.caller_items = c_items_.load();
    st.caller_calls = c_calls_.load();
    return st;
}

// port of main.rs: physical_cores (Windows: GetLogicalProcessorInformationEx, one entry per physical core).
size_t physical_cores() {
#ifdef _WIN32
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
    if (len == 0) return 0;
    std::vector<uint8_t> buf(len);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore, (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buf.data(), &len))
        return 0;
    size_t n = 0;
    for (DWORD off = 0; off < len;) {
        auto* info = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buf.data() + off);
        if (info->Relationship == RelationProcessorCore) n++;
        if (info->Size == 0) break;
        off += info->Size;
    }
    return n;
#else
    return 0;
#endif
}

size_t default_threads() {
    const size_t p = physical_cores();
    if (p > 0) return p;
    const unsigned hw = std::thread::hardware_concurrency();
    return hw ? hw : 1;
}

// ---------------------------------------------------------------- activations

std::vector<Q8K> quantize_q8k(const float* x, size_t n) {
    std::vector<Q8K> out;
    out.reserve((n + 255) / 256);
    for (size_t off = 0; off < n; off += 256) {
        float b[256] = {};
        const size_t len = std::min<size_t>(256, n - off);
        std::memcpy(b, x + off, len * sizeof(float));
        // The first value of largest magnitude (a max over |v| first, which vectorizes).
        float amax = 0.0f;
        for (float v : b) amax = std::max(amax, std::fabs(v));
        Q8K q{};
        if (amax == 0.0f) {
            q.d = 0.0f;
            out.push_back(q);
            continue;
        }
        float max = 0.0f;
        for (float v : b)
            if (std::fabs(v) == amax) {
                max = v;
                break;
            }
        const float is = -127.0f / max;
#if SC_AVX2
        quant_block_dispatch(is, b, q.qs);
#else
        for (size_t i = 0; i < 256; i++) {
            const float r = std::nearbyint(is * b[i]);
            q.qs[i] = (int8_t)std::min<int32_t>((int32_t)r, 127);
        }
#endif
        for (size_t g = 0; g < 16; g++) {
            int32_t s = 0;
            for (size_t k = 0; k < 16; k++) s += q.qs[16 * g + k];
            q.bs[g] = (int16_t)s;
        }
        q.d = 1.0f / is;
        out.push_back(q);
    }
    return out;
}

// ---------------------------------------------------------------- weights (model.rs: Model::w.layer)

bool load_expert_layer(const Weights& w, const ModelConfig& cfg, size_t il, ExpertLayer& out, std::string& err) {
    if (cfg.n_expert == 0) {
        err = "model has no routed experts";
        return false;
    }
    out.n_embd = cfg.n_embd;
    out.n_expert = cfg.n_expert;
    out.n_ff = cfg.n_ff_exp;
    // StrataFreet addition (W2): the Situ betas of inkling2 (0 keeps the SwiGLU activation of the qwen families).
    out.situ_beta = cfg.situ_beta;
    out.situ_linear_beta = cfg.situ_linear_beta;
    const size_t ne = cfg.n_expert, ff = cfg.n_ff_exp, nd = cfg.n_embd;
    // StrataFreet addition (INKLING2_NOTES.md): inkling2 fuses gate and up in one tensor. gate_scale /
    // down_scale ({n_expert} f32, one per expert per matrix) exist only on the quantized blocks.
    Tensor fused;
    if (w.layer(il, "ffn_gate_up_exps.weight", fused, err)) {
        std::string probe;
        if (w.layer(il, "ffn_gate_exps.weight", out.gate, probe) || w.layer(il, "ffn_up_exps.weight", out.up, probe)) {
            err = "blk." + std::to_string(il) + ": both fused and separate gate/up expert tensors";
            return false;
        }
        err.clear();
        out.fused_gu = true;
        out.gate_up = fused;
        if (fused.cols != nd || fused.rows != 2 * ff * ne) {
            err = "blk." + std::to_string(il) +
                  ".ffn_gate_up_exps: shape does not match n_embd/n_ff_exp/n_expert";
            return false;
        }
        if (!w.layer(il, "ffn_down_exps.weight", out.down, err)) return false;
        if (out.down.cols != ff || out.down.rows != nd * ne) {
            err = "blk." + std::to_string(il) + ".ffn_down_exps: shape does not match n_ff_exp/n_embd/n_expert";
            return false;
        }
        if (w.layer(il, "ffn_gate_up_exps.scale", out.gate_scale, probe) &&
            w.layer(il, "ffn_down_exps.scale", out.down_scale, probe)) {
            if (out.gate_scale.len != 4 * ne || out.down_scale.len != 4 * ne) {
                err = "blk." + std::to_string(il) + ".ffn_*_exps.scale: expected one f32 per expert";
                return false;
            }
            out.has_scales = true;
        } else {
            err.clear();
        }
        return true;
    }
    if (!w.layer(il, "ffn_gate_exps.weight", out.gate, err)) return false;
    if (!w.layer(il, "ffn_up_exps.weight", out.up, err)) return false;
    if (!w.layer(il, "ffn_down_exps.weight", out.down, err)) return false;
    if (out.gate.cols != nd || out.up.cols != nd || out.gate.rows != ff * ne || out.up.rows != ff * ne) {
        err = "blk." + std::to_string(il) + ".ffn_gate/up_exps: shape does not match n_embd/n_ff_exp/n_expert";
        return false;
    }
    if (out.down.cols != ff || out.down.rows != nd * ne) {
        err = "blk." + std::to_string(il) + ".ffn_down_exps: shape does not match n_ff_exp/n_embd/n_expert";
        return false;
    }
    return true;
}

// ---------------------------------------------------------------- experts (cpu_moe.rs: experts, experts_multi)

thread_local ExpertsTrace* g_experts_trace = nullptr;

// port of cpu_moe.rs: experts
bool experts(const ExpertLayer& L, const float* h, const std::vector<std::pair<size_t, float>>& sel, float* y,
             ThreadPool& pool, std::string& err) {
    ExpertsTrace* tr = g_experts_trace;   // STRATA_TRACE_WORKER: pass boundaries (see cpu_moe.h)
    auto now = [] {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
    };
    if (tr) tr->t[0] = now();
    const size_t n = L.n_embd;
    std::fill(y, y + n, 0.0f);
    if (sel.empty()) return true;
    const size_t k = sel.size(), ff = L.n_ff, ne = L.n_expert;
    for (const auto& s : sel)
        if (s.first >= ne) {
            err = "expert id " + std::to_string(s.first) + " out of range";
            return false;
        }
    // Gate and up of every expert in one parallel pass (gate and up share a type in all supported files).
    // StrataFreet addition: the fused gate_up of inkling2 hands out per-expert views — the expert's
    // slice holds its gate rows first, then its up rows.
    auto gate_up_of = [&](const ExpertLayer& l, size_t e, Tensor& g, Tensor& u) {
        if (!l.fused_gu) {
            g = l.gate.expert(e, ne);
            u = l.up.expert(e, ne);
            return;
        }
        const Tensor t = l.gate_up.expert(e, ne);
        g = t;
        g.rows = ff;
        g.len = ff * t.row_bytes();
        u = t;
        u.data += ff * t.row_bytes();
        u.rows = ff;
        u.len = ff * t.row_bytes();
    };
    // StrataFreet addition: the per-expert f32 scales of the quantized blocks (one per expert per
    // matrix, when the file has them), read straight from the mapped f32 rows.
    auto scale_of = [&](const ExpertLayer& l, const Tensor& s, size_t e) -> float {
        if (!l.has_scales) return 1.0f;
        float v;
        std::memcpy(&v, s.data + 4 * e, sizeof v);
        return v;
    };
    std::vector<Tensor> gu;
    gu.reserve(2 * k);
    for (const auto& s : sel) {
        Tensor g, u;
        gate_up_of(L, s.first, g, u);
        gu.push_back(g);
        gu.push_back(u);
    }
    std::vector<float> a(2 * k * ff);
    if (gu[0].ty == gu[1].ty) {
        batch_rows(gu, {h}, n, a.data(), 2 * k * ff, pool);
    } else {
        for (size_t i = 0; i < gu.size(); i++) matvec(gu[i], h, n, a.data() + i * ff, pool);
    }
    if (tr) tr->t[1] = now();
    std::vector<std::vector<float>> acts(k, std::vector<float>(ff));
    for (size_t j = 0; j < k; j++) {
        const float sg = scale_of(L, L.gate_scale, sel[j].first);
        const float* g = a.data() + 2 * j * ff;
        const float* u = g + ff;
        if (L.situ_beta > 0.0f) {
            // StrataFreet addition (W2): inkling2's Situ activation, matching llama.cpp's LLM_FFN_SITU branch:
            // situ(gate, up) = beta * tanh(gate/beta) * sigmoid(gate) * lb * tanh(up/lb).
            const float beta = L.situ_beta, lb = L.situ_linear_beta;
            for (size_t i = 0; i < ff; i++) {
                const float gv = g[i] * sg, uv = u[i] * sg;
                const float act = beta * std::tanh(gv / beta) * (1.0f / (1.0f + std::exp(-gv)));  // sigmoid(gate), not silu
                const float lin = lb > 0.0f ? lb * std::tanh(uv / lb) : uv;
                acts[j][i] = act * lin;
            }
        } else {
            for (size_t i = 0; i < ff; i++) acts[j][i] = silu(g[i] * sg) * (u[i] * sg);
        }
    }
    if (tr) tr->t[2] = now();
    // Down of every expert, over its activation.
    std::vector<Tensor> ds;
    ds.reserve(k);
    for (const auto& s : sel) ds.push_back(L.down.expert(s.first, ne));
    std::vector<const float*> xs;
    xs.reserve(k);
    for (size_t j = 0; j < k; j++) xs.push_back(acts[j].data());
    std::vector<float> o(k * n);
    batch_rows(ds, xs, ff, o.data(), k * n, pool);
    if (tr) tr->t[3] = now();
    for (size_t j = 0; j < k; j++) {
        const float w = sel[j].second * scale_of(L, L.down_scale, sel[j].first);
        for (size_t i = 0; i < n; i++) y[i] += w * o[j * n + i];
    }
    if (tr) tr->t[4] = now();
    return true;
}

// port of cpu_moe.rs: experts_multi
bool experts_multi(const ExpertLayer& L, const std::vector<std::vector<float>>& hs,
                   const std::vector<std::vector<std::pair<size_t, float>>>& sels, std::vector<std::vector<float>>& ys,
                   ThreadPool& pool, std::string& err) {
    const size_t nt = hs.size(), n = L.n_embd, ff = L.n_ff, ne = L.n_expert;
    ys.assign(nt, std::vector<float>(n, 0.0f));
    if (nt == 0) return true;
    if (sels.size() != nt) {
        err = "experts_multi: one selection per token is required";
        return false;
    }
    for (const auto& h : hs)
        if (h.size() != n) {
            err = "experts_multi: activation length is not n_embd";
            return false;
        }
    // BTreeMap in Rust: experts visited in id order, each with the tokens that routed to it (in token order).
    std::map<size_t, std::vector<std::pair<size_t, float>>> uses;
    for (size_t ti = 0; ti < nt; ti++)
        for (const auto& s : sels[ti]) {
            if (s.first >= ne) {
                err = "expert id " + std::to_string(s.first) + " out of range";
                return false;
            }
            uses[s.first].push_back({ti, s.second});
        }
    // Gate and up of every expert in one pass over its weights, for all tokens that route to it.
    // StrataFreet addition: the fused gate_up of inkling2 slices per expert (gate rows first, then up),
    // and the per-expert f32 scales multiply the dot outputs exactly as experts() does.
    using PerTok = std::vector<std::pair<size_t, std::vector<float>>>;
    auto gate_up_of = [&](const ExpertLayer& l, size_t e, Tensor& g, Tensor& u) {
        if (!l.fused_gu) {
            g = l.gate.expert(e, ne);
            u = l.up.expert(e, ne);
            return;
        }
        const Tensor t = l.gate_up.expert(e, ne);
        g = t;
        g.rows = ff;
        g.len = ff * t.row_bytes();
        u = t;
        u.data += ff * t.row_bytes();
        u.rows = ff;
        u.len = ff * t.row_bytes();
    };
    auto scale_of = [&](const ExpertLayer& l, const Tensor& s, size_t e) -> float {
        if (!l.has_scales) return 1.0f;
        float v;
        std::memcpy(&v, s.data + 4 * e, sizeof v);
        return v;
    };
    std::map<size_t, PerTok> acts;
    for (const auto& kv : uses) {
        const size_t e = kv.first;
        const auto& list = kv.second;
        std::vector<const float*> xs;
        for (const auto& p : list) xs.push_back(hs[p.first].data());
        Tensor gt, ut;
        gate_up_of(L, e, gt, ut);
        auto g = rows_multi(gt, xs, n, pool);
        auto u = rows_multi(ut, xs, n, pool);
        const float sg = scale_of(L, L.gate_scale, e);
        const float beta = L.situ_beta, lb = L.situ_linear_beta;
        PerTok a;
        for (size_t j = 0; j < list.size(); j++) {
            std::vector<float> v(ff);
            if (beta > 0.0f) {
                // StrataFreet addition (W2): the Situ activation (LLM_FFN_SITU).
                for (size_t i = 0; i < ff; i++) {
                    const float gv = g[j][i] * sg, uv = u[j][i] * sg;
                    const float act = beta * std::tanh(gv / beta) * (1.0f / (1.0f + std::exp(-gv)));  // sigmoid(gate), not silu
                    const float lin = lb > 0.0f ? lb * std::tanh(uv / lb) : uv;
                    v[i] = act * lin;
                }
            } else {
                for (size_t i = 0; i < ff; i++) v[i] = silu(g[j][i] * sg) * (u[j][i] * sg);
            }
            a.push_back({list[j].first, std::move(v)});
        }
        acts[e] = std::move(a);
    }
    // Down of every expert, over the acts of its tokens.
    std::map<size_t, PerTok> outs;
    for (const auto& kv : uses) {
        const size_t e = kv.first;
        const auto& list = kv.second;
        const PerTok& a = acts[e];
        std::vector<const float*> xs;
        for (const auto& p : a) xs.push_back(p.second.data());
        auto o = rows_multi(L.down.expert(e, ne), xs, ff, pool);
        const float sd = scale_of(L, L.down_scale, e);
        PerTok ov;
        for (size_t j = 0; j < list.size(); j++) {
            if (sd != 1.0f)
                for (size_t i = 0; i < n; i++) o[j][i] *= sd;
            ov.push_back({list[j].first, std::move(o[j])});
        }
        outs[e] = std::move(ov);
    }
    // The per-token sum keeps the router's expert order, so outputs match `experts`.
    for (size_t ti = 0; ti < nt; ti++)
        for (const auto& s : sels[ti]) {
            auto it = outs.find(s.first);
            if (it == outs.end()) continue;
            const std::vector<float>* o = nullptr;
            for (const auto& p : it->second)
                if (p.first == ti) {
                    o = &p.second;
                    break;
                }
            if (!o) continue;
            for (size_t k = 0; k < n; k++) ys[ti][k] += s.second * (*o)[k];
        }
    return true;
}

// port of cpu_moe.rs: int_dots_match_dequant (unit test). Every integer kernel, scalar and (when built) AVX2,
// against the f64 dot of the dequantized row with the Q8_K activation.
bool int_dots_selftest(std::string& err) {
    uint64_t r = 0x9E3779B97F4A7C15ULL;
    auto rnd = [&]() -> uint8_t {
        r ^= r << 13;
        r ^= r >> 7;
        r ^= r << 17;
        return (uint8_t)r;
    };
    float x[1024];
    for (int i = 0; i < 1024; i++) {
        const float f = (float)i * 0.7548777f;
        x[i] = ((f - std::floor(f)) - 0.5f) * 2.0f;   // Rust: (i as f32 * 0.7548777).fract()
    }
    // f16 0.01 (bits 0x211F, the nearest f16 to 0.01, as f16::from_f32 gives).
    const uint16_t f16_001 = 0x211F;
    const Ty tys[] = {Ty::Q2_0, Ty::Q4_0, Ty::Q4_1, Ty::Q5_0, Ty::Q5_1, Ty::IQ2XS, Ty::IQ3XXS, Ty::IQ3S, Ty::IQ4XS,
                      Ty::Q6K, Ty::Q2K, Ty::Q3K, Ty::Q4K, Ty::Q5K, Ty::Q8_0, Ty::IQ2XXS, Ty::IQ2S, Ty::IQ1S, Ty::IQ1M,
                      Ty::IQ4NL, Ty::MXFP4, Ty::TQ1_0, Ty::TQ2_0};
    for (Ty ty : tys) {
        const size_t n = ty == Ty::Q2_0 ? 640 : ty == Ty::IQ4NL ? 672 : 1024;   // Q2_0 exercises the zero-padded block
        const std::vector<Q8K> xq = quantize_q8k(x, n);
        const size_t be = block_elems(ty), bb = block_bytes(ty);
        std::vector<uint8_t> row(n / be * bb);
        for (auto& v : row) v = rnd();
        // Offsets of the f16 scales (d, and dmin where the type has one).
        std::vector<size_t> fs;
        switch (ty) {
            case Ty::Q6K: fs = {208}; break;
            case Ty::Q2K: fs = {80, 82}; break;
            case Ty::Q3K: fs = {108}; break;
            case Ty::Q4K: case Ty::Q5K: case Ty::Q4_1: case Ty::Q5_1: fs = {0, 2}; break;
            case Ty::IQ1M: break;
            case Ty::MXFP4: break;   // the scale is the E8M0 byte 0 of each block, set below
            default: fs = {0}; break;
        }
        for (size_t bi = 0; bi < row.size() / bb; bi++)
            for (size_t o : fs) std::memcpy(row.data() + bi * bb + o, &f16_001, 2);
        // MXFP4: E8M0 scales 2^-8 .. 2^7 (byte 120 + block index mod 16), so the scales vary.
        if (ty == Ty::MXFP4)
            for (size_t bi = 0; bi < row.size() / 17; bi++) row[bi * 17] = (uint8_t)(120 + bi % 16);
        // IQ1_M keeps d in the top nibbles of its four scale words.
        if (ty == Ty::IQ1M) {
            for (size_t bi = 0; bi < row.size() / 56; bi++)
                for (int i = 0; i < 4; i++) {
                    uint8_t& v = row[bi * 56 + 49 + 2 * i];
                    v = (uint8_t)((v & 0xF) | (((f16_001 >> (4 * i)) & 0xF) << 4));
                }
        }
        std::vector<float> wv(n);
        if (!dequant_row(ty, row.data(), row.size(), wv.data(), n)) {
            err = std::string(type_name(ty)) + ": dequant_row refused the row";
            return false;
        }
        double want = 0.0;
        for (size_t i = 0; i < n; i++) {
            const float qv = xq[i / 256].d * (float)xq[i / 256].qs[i % 256];
            want += (double)wv[i] * (double)qv;
        }
        const DotFn sc = scalar_dot(ty);
        const DotFn ic = int_dot(ty, false);
        if (!sc || !ic) {
            err = std::string(type_name(ty)) + ": no integer kernel";
            return false;
        }
        const float got_s = sc(row.data(), row.size(), xq.data(), xq.size());
        const float got_i = ic(row.data(), row.size(), xq.data(), xq.size());
        const float gots[2] = {got_s, got_i};
        const char* which[2] = {"scalar", "avx2"};
        for (int k = 0; k < 2; k++) {
            const double rel = ((double)gots[k] - want) / std::max(std::fabs(want), 1e-3);
            if (!(std::fabs(rel) < 1e-5)) {
                char buf[256];
                std::snprintf(buf, sizeof buf, "%s %s: got %.9g want %.9g", type_name(ty).c_str(), which[k], (double)gots[k], want);
                err = buf;
                return false;
            }
        }
    }
    return true;
}

// StrataFreet test hook (not in cpu_moe.rs): y[r] = (row r of w) . x for `rows` rows of `row_bytes` bytes of type
// `ty`, through the integer kernel (the scalar one when `scalar`, else the AVX2 one when built), on the Q8_K
// quantization of x. tests/test_cpu_moe.py compares it with a float64 dot of the dequantized rows.
bool int_dot_rows(Ty ty, bool scalar, const uint8_t* w, size_t rows, size_t rb, const float* x, size_t n, float* y,
                  std::string& err) {
    const DotFn dot = int_dot(ty, scalar);
    if (!dot) {
        err = std::string(type_name(ty)) + ": no integer kernel";
        return false;
    }
    const std::vector<Q8K> xq = quantize_q8k(x, n);
    for (size_t r = 0; r < rows; r++) y[r] = dot(w + r * rb, rb, xq.data(), xq.size());
    return true;
}

// port of cpu_moe.rs: int_dot (the public entry deepseek/mat.rs uses; STRATA_SCALAR read per call, as in Rust).
IntDotFn int_dot_kernel(Ty ty) { return int_dot(ty, scalar_env()); }

}  // namespace strata
