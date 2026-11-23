// port of backend.rs: upload_type, the Backend::upload default, and the cuda/mod.rs type policy Gpu::mmq.
//
// Part of strata-cpp, a C++ port of RustingStrata by Vasyl Trefilov (RustingStrata License 1.0, see
// LICENSE-RustingStrata.md).
#include "backend.h"

#include <cstdlib>
#include <cstring>

namespace strata {

// port of backend.rs: upload_type
Ty upload_type(Ty ty) {
    switch (ty) {
        case Ty::F16: return Ty::F32;
        case Ty::Q4_0:
        case Ty::Q5_0: return Ty::Q8_0;
        default: return ty;
    }
}

// port of backend.rs: Backend::upload (default body)
bool Backend::upload(const Tensor& t, GpuTensor& out, std::string& err) {
    if (t.ty == Ty::F16) {
        // F32 holds every F16 value exactly, and there are no F16 kernels.
        std::vector<float> f(t.rows * t.cols);
        if (!dequant_row(Ty::F16, t.data, t.len, f.data(), f.size())) {
            err = "upload: F16 tensor of " + std::to_string(t.len) + " bytes does not hold " +
                  std::to_string(t.rows * t.cols) + " values";
            return false;
        }
        Buf b;
        if (!alloc(4 * f.size(), b, err) || !htod(f.data(), 4 * f.size(), b, 0, err)) return false;
        out = GpuTensor(Ty::F32, t.rows, t.cols, std::move(b));
        return true;
    }
    if ((t.ty == Ty::Q4_0 || t.ty == Ty::Q5_0)) {
        // Q4_0/Q5_0 hold their values exactly as Q8_0 (to_q8_0 keeps the scales).
        const size_t blocks = t.len / block_bytes(t.ty);
        std::vector<uint8_t> q(blocks * 34);
        if (!to_q8_0(t.ty, t.data, t.len, q.data(), q.size())) {
            err = "upload: to_q8_0 refused the tensor bytes";
            return false;
        }
        Buf b;
        if (!alloc(q.size(), b, err) || !htod(q.data(), q.size(), b, 0, err)) return false;
        out = GpuTensor(Ty::Q8_0, t.rows, t.cols, std::move(b));
        return true;
    }
    Buf b;
    if (!alloc(t.len, b, err) || !htod(t.data, t.len, b, 0, err)) return false;
    out = GpuTensor(t.ty, t.rows, t.cols, std::move(b));
    return true;
}

// port of cuda/mod.rs: Gpu::mmq (the type policy, STRATA_MMQ overrides the default list)
bool cuda_mmq_policy(Ty ty) {
    // float_ty: F32 and BF16 never take the int8 path.
    if (ty == Ty::F32 || ty == Ty::BF16) return false;
    if (const char* v = std::getenv("STRATA_MMQ")) {
        // split(',').any(|t| t == ty.name())
        const char* name = ty_name(ty);
        std::string s(v);
        size_t i = 0;
        while (i <= s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            if (s.compare(i, j - i, name) == 0 && j - i == std::strlen(name)) return true;
            i = j + 1;
        }
        return false;
    }
    // IQ2_XS (and IQ2_S) has a scale per 16: its int8 path loses to f16 (#31) unless llama numerics is on.
    const bool iq2 = ty == Ty::IQ2XS || ty == Ty::IQ2S;
    return !iq2 || std::getenv("STRATA_LLAMA_NUMERICS") != nullptr;
}

}  // namespace strata
