// port of quant/mod.rs. Function names match the Rust ones so a reader can diff them side by side.
#include "quant.h"

#include "tables_host.h"

#include <cmath>
#include <cstring>

namespace strata {

using tables::IQ1S_GRID;
using tables::IQ2S_GRID;
using tables::IQ2XS_GRID;
using tables::IQ2XXS_GRID;
using tables::IQ3S_GRID;
using tables::IQ3XXS_GRID;
using tables::KMASK_IQ2XS;
using tables::KSIGNS_IQ2XS;
using tables::KVALUES_IQ4NL;

namespace {

inline uint16_t rd16(const uint8_t* b) { return (uint16_t)(b[0] | (b[1] << 8)); }

inline uint32_t rd32(const uint8_t* b) {
    return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
}

// port of half::f16::to_f32 (round-to-nearest-even is not needed: this is the exact widening conversion)
float h2f(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {  // subnormal half: normalize into a float mantissa
            int e = -1;
            do {
                e++;
                man <<= 1;
            } while ((man & 0x400u) == 0);
            man &= 0x3FFu;
            bits = sign | (uint32_t)(127 - 15 - e) << 23 | man << 13;
        }
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | man << 13;
    } else {
        bits = sign | (exp + (127 - 15)) << 23 | man << 13;
    }
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// port of quant/mod.rs: h (f16 at byte offset o)
inline float h(const uint8_t* b, size_t o) { return h2f(rd16(b + o)); }
// port of quant/mod.rs: bf16::from_le_bytes(..).to_f32()
inline float bf(const uint8_t* b, size_t o) {
    uint32_t bits = (uint32_t)rd16(b + o) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

// port of quant/mod.rs: scale_min_k4
inline void scale_min_k4(size_t j, const uint8_t* q, uint8_t& s, uint8_t& m) {
    if (j < 4) {
        s = (uint8_t)(q[j] & 63);
        m = (uint8_t)(q[j + 4] & 63);
    } else {
        s = (uint8_t)((q[j + 4] & 0xF) | ((uint8_t)(q[j - 4] >> 6) << 4));
        m = (uint8_t)((q[j + 4] >> 4) | ((uint8_t)(q[j] >> 6) << 4));
    }
}

// port of quant/mod.rs: tq1_trit
inline float tq1_trit(uint8_t v, size_t n) {
    static const uint8_t POW3[5] = {1, 3, 9, 27, 81};
    const uint8_t q = (uint8_t)(v * POW3[n]);
    return (float)(((uint16_t)((uint16_t)q * 3) >> 8)) - 1.0f;
}

// port of quant/mod.rs: tq1_0
void tq1_0(const uint8_t* b, float* y) {
    const float d = h(b, 52);
    size_t o = 0;
    for (size_t n = 0; n < 5; n++)
        for (size_t m = 0; m < 32; m++) y[o++] = d * tq1_trit(b[m], n);
    for (size_t n = 0; n < 5; n++)
        for (size_t m = 0; m < 16; m++) y[o++] = d * tq1_trit(b[32 + m], n);
    for (size_t n = 0; n < 4; n++)
        for (size_t j = 0; j < 4; j++) y[o++] = d * tq1_trit(b[48 + j], n);
}

// port of quant/mod.rs: tq2_0
void tq2_0(const uint8_t* b, float* y) {
    const float d = h(b, 64);
    for (size_t j = 0; j < 2; j++)
        for (size_t l = 0; l < 4; l++)
            for (size_t m = 0; m < 32; m++)
                y[32 * (4 * j + l) + m] = d * ((float)((b[32 * j + m] >> (2 * l)) & 3) - 1.0f);
}

// port of quant/mod.rs: q2k
void q2k(const uint8_t* b, float* y) {
    const uint8_t* sc = b;
    const float d = h(b, 80), mn = h(b, 82);
    size_t q = 16, is = 0, o = 0;
    static const int shifts[4] = {0, 2, 4, 6};
    for (int rep = 0; rep < 2; rep++) {
        for (int si = 0; si < 4; si++) {
            for (int half = 0; half < 2; half++) {
                const uint8_t s = sc[is++];
                const float dl = d * (float)(s & 0xF), ml = mn * (float)(s >> 4);
                for (size_t l = 0; l < 16; l++)
                    y[o++] = dl * (float)((b[q + l + 16 * (size_t)half] >> shifts[si]) & 3) - ml;
            }
        }
        q += 32;
    }
}

// port of quant/mod.rs: q3k_scales
void q3k_scales(const uint8_t* b, int8_t out[16]) {
    uint32_t aux[4] = {0, 0, 0, 0};
    for (size_t i = 0; i < 3; i++) aux[i] = rd32(b + 96 + 4 * i);
    const uint32_t k1 = 0x03030303u, k2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
    aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
    aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
    aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 4; j++) out[4 * i + j] = (int8_t)((aux[i] >> (8 * j)) & 0xFF);
}

// port of quant/mod.rs: q3k
void q3k(const uint8_t* b, float* y) {
    const uint8_t* hm = b;
    const float d_all = h(b, 108);
    int8_t scales[16];
    q3k_scales(b, scales);
    size_t q = 32, is = 0, o = 0;
    uint8_t m = 1;
    static const int shifts[4] = {0, 2, 4, 6};
    for (int rep = 0; rep < 2; rep++) {
        for (int si = 0; si < 4; si++) {
            for (int half = 0; half < 2; half++) {
                const float dl = d_all * (float)((int)scales[is] - 32);
                is++;
                for (size_t l = 0; l < 16; l++) {
                    const size_t i = l + 16 * (size_t)half;
                    const int v = (int)((b[q + i] >> shifts[si]) & 3) - ((hm[i] & m) != 0 ? 0 : 4);
                    y[o++] = dl * (float)v;
                }
            }
            m = (uint8_t)(m << 1);
        }
        q += 32;
    }
}

// port of quant/mod.rs: q4k
void q4k(const uint8_t* b, float* y) {
    const float d = h(b, 0), mn = h(b, 2);
    const uint8_t* sc = b + 4;
    size_t q = 16, o = 0;
    for (size_t j = 0; j < 4; j++) {
        uint8_t s1, m1, s2, m2;
        scale_min_k4(2 * j, sc, s1, m1);
        scale_min_k4(2 * j + 1, sc, s2, m2);
        const float d1 = d * (float)s1, f1 = mn * (float)m1;
        const float d2 = d * (float)s2, f2 = mn * (float)m2;
        for (size_t l = 0; l < 32; l++) y[o + l] = d1 * (float)(b[q + l] & 0xF) - f1;
        for (size_t l = 0; l < 32; l++) y[o + 32 + l] = d2 * (float)(b[q + l] >> 4) - f2;
        q += 32;
        o += 64;
    }
}

// port of quant/mod.rs: q5k
void q5k(const uint8_t* b, float* y) {
    const float d = h(b, 0), mn = h(b, 2);
    const uint8_t* sc = b + 4;
    const uint8_t* qh = b + 16;
    size_t q = 48, o = 0;
    uint8_t u1 = 1, u2 = 2;
    for (size_t j = 0; j < 4; j++) {
        uint8_t s1, m1, s2, m2;
        scale_min_k4(2 * j, sc, s1, m1);
        scale_min_k4(2 * j + 1, sc, s2, m2);
        const float d1 = d * (float)s1, f1 = mn * (float)m1;
        const float d2 = d * (float)s2, f2 = mn * (float)m2;
        for (size_t l = 0; l < 32; l++)
            y[o + l] = d1 * ((float)(b[q + l] & 0xF) + ((qh[l] & u1) != 0 ? 16.0f : 0.0f)) - f1;
        for (size_t l = 0; l < 32; l++)
            y[o + 32 + l] = d2 * ((float)(b[q + l] >> 4) + ((qh[l] & u2) != 0 ? 16.0f : 0.0f)) - f2;
        q += 32;
        o += 64;
        u1 = (uint8_t)(u1 << 2);
        u2 = (uint8_t)(u2 << 2);
    }
}

// port of quant/mod.rs: q6k
void q6k(const uint8_t* b, float* y) {
    const float d = h(b, 208);
    for (size_t n = 0; n < 2; n++) {
        const uint8_t* ql = b + 64 * n;
        const uint8_t* qh = b + 128 + 32 * n;
        const uint8_t* sc = b + 192 + 8 * n;
        const size_t yo = 128 * n;
        for (size_t l = 0; l < 32; l++) {
            const size_t is = l / 16;
            const int q1 = (int)((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
            const int q2 = (int)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int q3 = (int)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int q4 = (int)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            y[yo + l] = d * (float)(int8_t)sc[is] * (float)q1;
            y[yo + l + 32] = d * (float)(int8_t)sc[is + 2] * (float)q2;
            y[yo + l + 64] = d * (float)(int8_t)sc[is + 4] * (float)q3;
            y[yo + l + 96] = d * (float)(int8_t)sc[is + 6] * (float)q4;
        }
    }
}

// port of quant/mod.rs: sign
inline float sign8(uint8_t signs, size_t j) { return (signs & KMASK_IQ2XS[j]) != 0 ? -1.0f : 1.0f; }

// port of quant/mod.rs: iq1s
void iq1s(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    for (size_t ib = 0; ib < 8; ib++) {
        const size_t qh = (size_t)rd16(b + 34 + 2 * ib);
        const float dl = d * (float)(2 * ((qh >> 12) & 7) + 1);
        const float delta = (qh & 0x8000) != 0 ? -0.125f : 0.125f;
        for (size_t l = 0; l < 4; l++) {
            const uint8_t* g = (const uint8_t*)&IQ1S_GRID[b[2 + 4 * ib + l] | (((qh >> (3 * l)) & 7) << 8)];
            for (size_t j = 0; j < 8; j++) y[32 * ib + 8 * l + j] = dl * ((float)(int8_t)g[j] + delta);
        }
    }
}

// port of quant/mod.rs: iq1m
void iq1m(const uint8_t* b, float* y) {
    size_t sc[4];
    for (size_t i = 0; i < 4; i++) sc[i] = (size_t)rd16(b + 48 + 2 * i);
    const float d = h2f((uint16_t)((sc[0] >> 12) | ((sc[1] >> 8) & 0xF0) | ((sc[2] >> 4) & 0xF00) |
                                   (sc[3] & 0xF000)));
    for (size_t ib = 0; ib < 8; ib++) {
        for (size_t l = 0; l < 4; l++) {
            const float dl = d * (float)(2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3 * (l / 2))) & 7) + 1);
            const size_t qh = (size_t)(b[32 + 2 * ib + l / 2] >> (4 * (l % 2)));
            const uint8_t* g = (const uint8_t*)&IQ1S_GRID[b[4 * ib + l] | ((qh & 7) << 8)];
            const float delta = (qh & 8) != 0 ? -0.125f : 0.125f;
            for (size_t j = 0; j < 8; j++) y[32 * ib + 8 * l + j] = dl * ((float)(int8_t)g[j] + delta);
        }
    }
}

// port of quant/mod.rs: iq2xxs
void iq2xxs(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    for (size_t ib = 0; ib < 8; ib++) {
        const uint8_t* qs = b + 2 + 8 * ib;
        const uint32_t aux = rd32(b + 6 + 8 * ib);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.25f;
        for (size_t l = 0; l < 4; l++) {
            const uint8_t* grid = (const uint8_t*)&IQ2XXS_GRID[qs[l]];
            const uint8_t signs = KSIGNS_IQ2XS[(aux >> (7 * l)) & 127];
            for (size_t j = 0; j < 8; j++) y[32 * ib + 8 * l + j] = db * (float)grid[j] * sign8(signs, j);
        }
    }
}

// port of quant/mod.rs: iq2s
void iq2s(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    const uint8_t* qs = b + 2;
    const uint8_t* signs = b + 34;
    const uint8_t* qh = b + 66;
    const uint8_t* scales = b + 74;
    for (size_t ib = 0; ib < 8; ib++) {
        const float db[2] = {d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f,
                             d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f};
        for (size_t l = 0; l < 4; l++) {
            const size_t gi = (size_t)qs[4 * ib + l] | ((((size_t)qh[ib] << (8 - 2 * l)) & 0x300));
            const uint8_t* grid = (const uint8_t*)&IQ2S_GRID[gi];
            for (size_t j = 0; j < 8; j++)
                y[32 * ib + 8 * l + j] = db[l / 2] * (float)grid[j] * sign8(signs[4 * ib + l], j);
        }
    }
}

// port of quant/mod.rs: iq2xs
void iq2xs(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    const uint8_t* scales = b + 66;
    for (size_t ib = 0; ib < 8; ib++) {
        const float db[2] = {d * (0.5f + (float)(scales[ib] & 0xF)) * 0.25f,
                             d * (0.5f + (float)(scales[ib] >> 4)) * 0.25f};
        for (size_t l = 0; l < 4; l++) {
            const size_t o = 2 + 2 * (4 * ib + l);
            const uint16_t q = rd16(b + o);
            const uint8_t* grid = (const uint8_t*)&IQ2XS_GRID[q & 511];
            const uint8_t signs = KSIGNS_IQ2XS[q >> 9];
            for (size_t j = 0; j < 8; j++)
                y[32 * ib + 8 * l + j] = db[l / 2] * (float)grid[j] * sign8(signs, j);
        }
    }
}

// port of quant/mod.rs: iq3xxs
void iq3xxs(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    const uint8_t* qs = b + 2;
    const uint8_t* ss = b + 66;
    for (size_t ib = 0; ib < 8; ib++) {
        const uint32_t aux = rd32(ss + 4 * ib);
        const float db = d * (0.5f + (float)(aux >> 28)) * 0.5f;
        for (size_t l = 0; l < 4; l++) {
            const uint8_t signs = KSIGNS_IQ2XS[(aux >> (7 * l)) & 127];
            const uint8_t* g1 = (const uint8_t*)&IQ3XXS_GRID[qs[8 * ib + 2 * l]];
            const uint8_t* g2 = (const uint8_t*)&IQ3XXS_GRID[qs[8 * ib + 2 * l + 1]];
            const size_t o = 32 * ib + 8 * l;
            for (size_t j = 0; j < 4; j++) {
                y[o + j] = db * (float)g1[j] * sign8(signs, j);
                y[o + j + 4] = db * (float)g2[j] * sign8(signs, j + 4);
            }
        }
    }
}

// port of quant/mod.rs: iq3s
void iq3s(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    const uint8_t* qs = b + 2;
    const uint8_t* qh = b + 66;
    const uint8_t* signs = b + 74;
    const uint8_t* sc = b + 106;
    for (size_t ib = 0; ib < 8; ib++) {
        const float db = d * (float)(1 + 2 * ((sc[ib / 2] >> (4 * (ib % 2))) & 0xF));
        for (size_t l = 0; l < 4; l++) {
            const size_t hb = (size_t)qh[ib];
            const size_t i1 = (size_t)qs[8 * ib + 2 * l] | (((hb << (8 - 2 * l)) & 256));
            const size_t i2 = (size_t)qs[8 * ib + 2 * l + 1] | (((hb << (7 - 2 * l)) & 256));
            const uint8_t* g1 = (const uint8_t*)&IQ3S_GRID[i1];
            const uint8_t* g2 = (const uint8_t*)&IQ3S_GRID[i2];
            const uint8_t sg = signs[4 * ib + l];
            const size_t o = 32 * ib + 8 * l;
            for (size_t j = 0; j < 4; j++) {
                y[o + j] = db * (float)g1[j] * sign8(sg, j);
                y[o + j + 4] = db * (float)g2[j] * sign8(sg, j + 4);
            }
        }
    }
}

// port of quant/mod.rs: iq4xs
void iq4xs(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    const uint16_t sh = rd16(b + 2);
    const uint8_t* sl = b + 4;
    const uint8_t* qs = b + 8;
    for (size_t ib = 0; ib < 8; ib++) {
        const int ls = (int)((sl[ib / 2] >> (4 * (ib % 2))) & 0xF) | ((int)((sh >> (2 * ib)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        for (size_t j = 0; j < 16; j++) {
            const uint8_t q = qs[16 * ib + j];
            y[32 * ib + j] = dl * (float)KVALUES_IQ4NL[q & 0xF];
            y[32 * ib + j + 16] = dl * (float)KVALUES_IQ4NL[q >> 4];
        }
    }
}

// port of quant/mod.rs: q5_codes
void q5_codes(Ty ty, const uint8_t* b, int8_t q[32]) {
    uint32_t qh = 0;
    const uint8_t* qs;
    int off;
    if (ty == Ty::Q5_0) {
        qh = rd32(b + 2);
        qs = b + 6;
        off = 16;
    } else {
        qs = b + 2;
        off = 8;
    }
    for (size_t j = 0; j < 16; j++) {
        q[j] = (int8_t)((int)((qs[j] & 0xF) | (((qh >> j) & 1) << 4)) - off);
        q[j + 16] = (int8_t)((int)((qs[j] >> 4) | (((qh >> (j + 16)) & 1) << 4)) - off);
    }
}

// port of quant/mod.rs: iq4nl
void iq4nl(const uint8_t* b, float* y) {
    const float d = h(b, 0);
    for (size_t j = 0; j < 16; j++) {
        y[j] = d * (float)KVALUES_IQ4NL[b[2 + j] & 0xF];
        y[j + 16] = d * (float)KVALUES_IQ4NL[b[2 + j] >> 4];
    }
}

// StrataFreet addition (ggml block_mxfp4): 1 byte E8M0 scale, 16 bytes of E2M1 codes. Same values as the torch path
// of stratafreet/quant.py (kv[code] * 2^(e - 128)), which is the reference of tests/test_quant.py.
void mxfp4(const uint8_t* b, float* y) {
    const float d = e8m0_half(b[0]);
    for (size_t j = 0; j < 16; j++) {
        y[j] = d * (float)kKvaluesMxfp4[b[1 + j] & 0xF];
        y[j + 16] = d * (float)kKvaluesMxfp4[b[1 + j] >> 4];
    }
}

// StrataFreet addition (ggml-impl.h: ggml_ue4m3_to_fp32): UE4M3 (unsigned E4M3, bias 7) scale, returned
// halved so it pairs with the doubled kKvaluesMxfp4 exactly as ggml pairs them. x == 0 or 0x7F decodes
// to 0 (ggml rejects the NaN encoding instead of returning a value).
float ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = std::ldexp((float)man, -9);
    } else {
        raw = std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

// StrataFreet addition (ggml block_nvfp4): 64 elements = 4 sub-blocks of 16; d[4] UE4M3 scales (one per
// sub-block) + qs[32] packed E2M1 nibbles. Element layout per sub-block s (ggml-quants.c
// dequantize_row_nvfp4): yb[j] = v0*d, yb[j+8] = v1*d for j in 0..7, with v0/v1 the low/high nibble of
// byte qs[s*8 + j].
void nvfp4(const uint8_t* b, float* y) {
    for (size_t s = 0; s < 4; s++) {
        const float d = ue4m3_half(b[s]);
        const uint8_t* qs = b + 4 + s * 8;
        float* yb = y + s * 16;
        for (size_t j = 0; j < 8; j++) {
            yb[j] = d * (float)kKvaluesMxfp4[qs[j] & 0xF];
            yb[j + 8] = d * (float)kKvaluesMxfp4[qs[j] >> 4];
        }
    }
}

void dequant_block(Ty ty, const uint8_t* b, float* y) {
    switch (ty) {
        case Ty::F32: {
            uint32_t bits = rd32(b);
            std::memcpy(y, &bits, sizeof bits);
            return;
        }
        case Ty::F16: y[0] = h(b, 0); return;
        case Ty::BF16: y[0] = bf(b, 0); return;
        case Ty::Q2_0: {
            const float d = h(b, 0);
            for (size_t j = 0; j < 64; j++)
                y[j] = ((float)((b[2 + j / 4] >> (2 * (j % 4))) & 3) - 1.0f) * d;
            return;
        }
        case Ty::TQ1_0: tq1_0(b, y); return;
        case Ty::TQ2_0: tq2_0(b, y); return;
        case Ty::Q4_0:
        case Ty::Q5_0: {
            const float d = h(b, 0);
            int8_t q[32];
            q5_codes(ty, b, q);
            for (size_t j = 0; j < 32; j++) y[j] = (float)q[j] * d;
            return;
        }
        case Ty::Q4_1:
        case Ty::Q5_1: {
            const bool q51 = ty == Ty::Q5_1;
            const float d = h(b, 0), m = h(b, 2);
            const uint32_t qh = q51 ? rd32(b + 4) : 0u;
            const uint8_t* qs = q51 ? b + 8 : b + 4;
            for (size_t j = 0; j < 16; j++) {
                y[j] = d * (float)((uint32_t)(qs[j] & 0xF) | (q51 ? (((qh >> j) & 1) << 4) : 0u)) + m;
                y[j + 16] =
                    d * (float)((uint32_t)(qs[j] >> 4) | (q51 ? (((qh >> (j + 16)) & 1) << 4) : 0u)) + m;
            }
            return;
        }
        case Ty::Q8_0: {
            const float d = h(b, 0);
            for (size_t j = 0; j < 32; j++) y[j] = (float)(int8_t)b[2 + j] * d;
            return;
        }
        case Ty::Q2K: q2k(b, y); return;
        case Ty::Q3K: q3k(b, y); return;
        case Ty::Q4K: q4k(b, y); return;
        case Ty::Q5K: q5k(b, y); return;
        case Ty::Q6K: q6k(b, y); return;
        case Ty::IQ1S: iq1s(b, y); return;
        case Ty::IQ1M: iq1m(b, y); return;
        case Ty::IQ2XXS: iq2xxs(b, y); return;
        case Ty::IQ2XS: iq2xs(b, y); return;
        case Ty::IQ2S: iq2s(b, y); return;
        case Ty::IQ3XXS: iq3xxs(b, y); return;
        case Ty::IQ3S: iq3s(b, y); return;
        case Ty::IQ4XS: iq4xs(b, y); return;
        case Ty::IQ4NL: iq4nl(b, y); return;
        case Ty::MXFP4: mxfp4(b, y); return;
        case Ty::NVFP4: nvfp4(b, y); return;
    }
}

}  // namespace

// port of ggml_e8m0_to_fp32_half (ggml-impl.h): x < 2 gives the subnormal 0x00200000 << x, else the exponent x - 1.
float e8m0_half(uint8_t e) {
    const uint32_t bits = e < 2 ? (0x00200000u << e) : ((uint32_t)(e - 1) << 23);
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

std::optional<Ty> ty_from_u32(uint32_t id) {
    switch (id) {
        case 0: return Ty::F32;
        case 1: return Ty::F16;
        case 30: return Ty::BF16;
        case 42: return Ty::Q2_0;
        case 34: return Ty::TQ1_0;
        case 35: return Ty::TQ2_0;
        case 2: return Ty::Q4_0;
        case 3: return Ty::Q4_1;
        case 6: return Ty::Q5_0;
        case 7: return Ty::Q5_1;
        case 8: return Ty::Q8_0;
        case 10: return Ty::Q2K;
        case 11: return Ty::Q3K;
        case 12: return Ty::Q4K;
        case 13: return Ty::Q5K;
        case 14: return Ty::Q6K;
        case 16: return Ty::IQ2XXS;
        case 17: return Ty::IQ2XS;
        case 19: return Ty::IQ1S;
        case 29: return Ty::IQ1M;
        case 18: return Ty::IQ3XXS;
        case 21: return Ty::IQ3S;
        case 22: return Ty::IQ2S;
        case 23: return Ty::IQ4XS;
        case 20: return Ty::IQ4NL;
        case 39: return Ty::MXFP4;
        case 40: return Ty::NVFP4;
        default: return std::nullopt;
    }
}

size_t block_elems(Ty t) {
    switch (t) {
        case Ty::F32:
        case Ty::F16:
        case Ty::BF16: return 1;
        case Ty::Q4_0:
        case Ty::Q4_1:
        case Ty::Q5_0:
        case Ty::Q5_1:
        case Ty::Q8_0:
        case Ty::IQ4NL:
        case Ty::MXFP4: return 32;
        case Ty::Q2_0: return 64;
        case Ty::NVFP4: return 64;
        default: return 256;
    }
}

size_t block_bytes(Ty t) {
    switch (t) {
        case Ty::F32: return 4;
        case Ty::F16: return 2;
        case Ty::BF16: return 2;
        case Ty::Q2_0: return 18;
        case Ty::TQ1_0: return 54;
        case Ty::TQ2_0: return 66;
        case Ty::Q4_0: return 18;
        case Ty::Q4_1: return 20;
        case Ty::Q5_0: return 22;
        case Ty::Q5_1: return 24;
        case Ty::Q8_0: return 34;
        case Ty::Q2K: return 84;
        case Ty::Q3K: return 110;
        case Ty::Q4K: return 144;
        case Ty::Q5K: return 176;
        case Ty::Q6K: return 210;
        case Ty::IQ1S: return 50;
        case Ty::IQ1M: return 56;
        case Ty::IQ2XXS: return 66;
        case Ty::IQ2XS: return 74;
        case Ty::IQ2S: return 82;
        case Ty::IQ3XXS: return 98;
        case Ty::IQ3S: return 110;
        case Ty::IQ4XS: return 136;
        case Ty::IQ4NL: return 18;
        case Ty::MXFP4: return 17;
        case Ty::NVFP4: return 36;  // d[4] UE4M3 scales + qs[32] packed E2M1 nibbles
    }
    return 0;
}

size_t row_bytes(Ty t, uint64_t n) { return (size_t)(n / block_elems(t)) * block_bytes(t); }

const char* ty_name(Ty t) {
    switch (t) {
        case Ty::F32: return "F32";
        case Ty::F16: return "F16";
        case Ty::BF16: return "BF16";
        case Ty::Q2_0: return "Q2_0";
        case Ty::TQ1_0: return "TQ1_0";
        case Ty::TQ2_0: return "TQ2_0";
        case Ty::Q4_0: return "Q4_0";
        case Ty::Q4_1: return "Q4_1";
        case Ty::Q5_0: return "Q5_0";
        case Ty::Q5_1: return "Q5_1";
        case Ty::Q8_0: return "Q8_0";
        case Ty::Q2K: return "Q2_K";
        case Ty::Q3K: return "Q3_K";
        case Ty::Q4K: return "Q4_K";
        case Ty::Q5K: return "Q5_K";
        case Ty::Q6K: return "Q6_K";
        case Ty::IQ1S: return "IQ1_S";
        case Ty::IQ1M: return "IQ1_M";
        case Ty::IQ2XXS: return "IQ2_XXS";
        case Ty::IQ2XS: return "IQ2_XS";
        case Ty::IQ2S: return "IQ2_S";
        case Ty::IQ3XXS: return "IQ3_XXS";
        case Ty::IQ3S: return "IQ3_S";
        case Ty::IQ4XS: return "IQ4_XS";
        case Ty::IQ4NL: return "IQ4_NL";
        case Ty::MXFP4: return "MXFP4";
        case Ty::NVFP4: return "NVFP4";
    }
    return "?";
}

bool dequant_row(Ty ty, const uint8_t* src, size_t src_len, float* dst, size_t n) {
    const size_t be = block_elems(ty), bb = block_bytes(ty);
    if (be == 0 || bb == 0 || n % be != 0) return false;
    if (src_len < row_bytes(ty, n)) return false;
    for (size_t i = 0; i < n / be; i++) dequant_block(ty, src + i * bb, dst + i * be);
    return true;
}

bool to_q8_0(Ty ty, const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_len) {
    if (ty != Ty::Q4_0 && ty != Ty::Q5_0) return false;
    const size_t bb = block_bytes(ty);
    if (src_len % bb != 0) return false;
    const size_t blocks = src_len / bb;
    if (dst_len < blocks * 34) return false;
    for (size_t i = 0; i < blocks; i++) {
        const uint8_t* b = src + i * bb;
        uint8_t* o = dst + i * 34;
        int8_t q[32];
        q5_codes(ty, b, q);
        o[0] = b[0];
        o[1] = b[1];
        for (size_t j = 0; j < 32; j++) o[2 + j] = (uint8_t)q[j];
    }
    return true;
}

}  // namespace strata
