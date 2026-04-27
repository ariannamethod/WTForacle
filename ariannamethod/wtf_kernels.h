// wtf_kernels.h — thin shim over notorch + gguf for WTForacle inference.
//
// Two primitives only:
//   - wtf_dequant_to_f32: GGML quant blob → contiguous float32 row-major
//   - wtf_sgemv         : out[m] = W[m,n] @ x[n]  via Accelerate / OpenBLAS
//
// The full notorch.c / gguf.c are vendored alongside as source-of-truth;
// this header exposes only what the Go side calls through cgo.

#ifndef WTF_KERNELS_H
#define WTF_KERNELS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGML tensor type tags — mirror the constants in gguf.h.
#define WTF_DTYPE_F32  0
#define WTF_DTYPE_F16  1
#define WTF_DTYPE_Q4_0 2
#define WTF_DTYPE_Q5_0 6
#define WTF_DTYPE_Q8_0 8
#define WTF_DTYPE_Q4_K 12
#define WTF_DTYPE_Q6_K 14

// Dequantize n_elements from src (raw GGUF bytes for given dtype) into dst (f32).
// Returns 0 on success, -1 on unsupported dtype.
int wtf_dequant_to_f32(const uint8_t* src, int dtype, uint64_t n_elements, float* dst);

// out[m] = W[m,n] @ x[n]  — row-major. Routes to nt_blas_matvec when USE_BLAS,
// otherwise scalar fallback.
void wtf_sgemv(float* out, const float* W, const float* x, int m, int n);

// Strided sgemv with custom row stride (lda) and optional transpose.
//   trans = 0: out[m] = W[m,n] @ x[n]                       (NoTrans)
//   trans = 1: out[n] = W[m,n]^T @ x[m]                     (Trans)
// lda is the row stride of W (>= n). Used for KV-cache views where the
// logical [pos+1, head_dim] sub-matrix lives inside a wider [pos+1, kv_dim]
// buffer.
void wtf_sgemv_strided(float* out, const float* W, int lda,
                       const float* x, int m, int n, int trans);

#ifdef __cplusplus
}
#endif

#endif // WTF_KERNELS_H
