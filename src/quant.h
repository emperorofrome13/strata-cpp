// port of quant/mod.rs (GGML block layouts and scalar reference dequantization).
// Also ports the GgmlType helper impl block: block_elems, block_bytes, row_bytes, name.
//
// The block layouts and the arithmetic are llama.cpp's ggml-quants.c (MIT License, ggml authors),
// reached through RustingStrata 1.2.5. Every function below keeps the Rust function's name.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

namespace strata {

// port of quant/mod.rs: GgmlType
enum class Ty : uint32_t {
    F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
    Q2K = 10, Q3K = 11, Q4K = 12, Q5K = 13, Q6K = 14,
    IQ2XXS = 16, IQ2XS = 17, IQ3XXS = 18, IQ1S = 19, IQ4NL = 20, IQ3S = 21, IQ2S = 22, IQ4XS = 23,
    IQ1M = 29, BF16 = 30, TQ1_0 = 34, TQ2_0 = 35, MXFP4 = 39, NVFP4 = 40, Q2_0 = 42,
};

// StrataFreet addition (not in Strata 1.2.5; see MODIFICATIONS.md). ggml's kvalues_mxfp4: the E2M1 values doubled
// (0, 0.5, 1, 1.5, 2, 3, 4, 6 times 2), so the scale is halved by e8m0_half. Element j of a block is the table
// entry of the low nibble of byte j (j < 16), or of the high nibble of byte j - 16.
inline constexpr int8_t kKvaluesMxfp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
// StrataFreet addition: ggml_e8m0_to_fp32_half (ggml-impl.h). E8M0 byte e is 2^(e - 128), exact, including the
// subnormal scales 2^-128 and 2^-127 that the float bit pattern 0x00200000 << e gives.
float e8m0_half(uint8_t e);

// StrataFreet addition: ggml_ue4m3_to_fp32 (ggml-impl.h). UE4M3 (unsigned E4M3, bias 7) scale, returned halved so
// it pairs with the doubled kKvaluesMxfp4 (ggml's own convention). x == 0 or 0x7F decodes to 0.
float ue4m3_half(uint8_t x);

// port of quant/mod.rs: GgmlType::from_u32
std::optional<Ty> ty_from_u32(uint32_t id);
// port of quant/mod.rs: GgmlType::block_elems
size_t block_elems(Ty t);
// port of quant/mod.rs: GgmlType::block_bytes
size_t block_bytes(Ty t);
// port of quant/mod.rs: GgmlType::row_bytes
size_t row_bytes(Ty t, uint64_t n);
// port of quant/mod.rs: GgmlType::name
const char* ty_name(Ty t);

// port of quant/mod.rs: dequant_row. `dst` holds `n` elements; `src` must hold
// exactly row_bytes(ty, n) bytes. Returns false on a size mismatch (never reads past src).
bool dequant_row(Ty ty, const uint8_t* src, size_t src_len, float* dst, size_t n);

// port of quant/mod.rs: to_q8_0 (Q4_0/Q5_0 bytes as Q8_0 with the same scales).
bool to_q8_0(Ty ty, const uint8_t* src, size_t src_len, uint8_t* dst, size_t dst_len);

}  // namespace strata
