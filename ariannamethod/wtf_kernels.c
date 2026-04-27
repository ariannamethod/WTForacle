// wtf_kernels.c — thin shim over notorch + gguf for WTForacle inference.
//
// Public surface: wtf_dequant_to_f32 + wtf_sgemv (declared in wtf_kernels.h).
//
// Dequant kernels (Q4_0 / Q5_0 / Q8_0 / Q4_K / Q6_K / F16) are implemented
// here directly to keep vendored notorch source untouched — gguf.c keeps
// these helpers static, so we re-spell them rather than patch the vendor.
// The math is bit-identical to notorch/gguf.c.
//
// SGEMV is delegated to nt_blas_matvec (notorch.c), which routes to
// cblas_sgemv via Apple Accelerate / OpenBLAS when USE_BLAS is defined.

#include "wtf_kernels.h"
#include "notorch.h"
#include <stdio.h>
#include <string.h>

// ── F16 → F32 ───────────────────────────────────────────────────────────────
static float wtf_f16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1u;
    uint32_t exp  = (h >> 10) & 0x1Fu;
    uint32_t mant = h & 0x3FFu;
    if (exp == 0) {
        if (mant == 0) {
            uint32_t r = sign << 31; float f; memcpy(&f, &r, 4); return f;
        }
        while (!(mant & 0x400u)) { mant <<= 1; exp--; }
        exp++; mant &= ~0x400u;
    } else if (exp == 31) {
        uint32_t r = (sign << 31) | 0x7F800000u | (mant << 13);
        float f; memcpy(&f, &r, 4); return f;
    }
    exp = exp + 127u - 15u;
    uint32_t r = (sign << 31) | (exp << 23) | (mant << 13);
    float f; memcpy(&f, &r, 4);
    return f;
}

// ── Dequant kernels — match notorch/gguf.c byte-for-byte ────────────────────

static void deq_q4_0(const uint8_t* src, float* dst, uint64_t n) {
    uint64_t nb = n / 32;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 18;
        uint16_t sh; memcpy(&sh, blk, 2);
        float scale = wtf_f16_to_f32(sh);
        for (int i = 0; i < 16; i++) {
            uint8_t byte = blk[2 + i];
            int lo = (int)(byte & 0x0Fu) - 8;
            int hi = (int)(byte >> 4)    - 8;
            dst[b * 32 + i]      = (float)lo * scale;
            dst[b * 32 + i + 16] = (float)hi * scale;
        }
    }
}

static void deq_q8_0(const uint8_t* src, float* dst, uint64_t n) {
    uint64_t nb = n / 32;
    for (uint64_t b = 0; b < nb; b++) {
        const uint8_t* blk = src + b * 34;
        uint16_t sh; memcpy(&sh, blk, 2);
        float scale = wtf_f16_to_f32(sh);
        for (int i = 0; i < 32; i++) {
            dst[b * 32 + i] = (float)(int8_t)blk[2 + i] * scale;
        }
    }
}

static void deq_f16(const uint8_t* src, float* dst, uint64_t n) {
    const uint16_t* p = (const uint16_t*)src;
    for (uint64_t i = 0; i < n; i++) dst[i] = wtf_f16_to_f32(p[i]);
}

static void get_scale_min_k4(int j, const uint8_t* sc, uint8_t* s, uint8_t* m) {
    if (j < 4) { *s = sc[j] & 63; *m = sc[j+4] & 63; }
    else { *s = (sc[j+4] & 0x0F) | ((sc[j-4] >> 6) << 4);
           *m = (sc[j+4] >> 4)   | ((sc[j]   >> 6) << 4); }
}

static void deq_q4_k(const uint8_t* src, float* dst, uint64_t n) {
    uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t* b = src + i * 144;
        float d    = wtf_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
        float dmin = wtf_f16_to_f32((uint16_t)(b[2] | (b[3] << 8)));
        const uint8_t *sc = b + 4, *qs = b + 16;
        int is = 0, qi = 0; uint64_t oi = i * 256;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc0, m0, sc1, m1v;
            get_scale_min_k4(is,   sc, &sc0, &m0);
            get_scale_min_k4(is+1, sc, &sc1, &m1v);
            float d1 = d * (float)sc0,  mm1 = dmin * (float)m0;
            float d2 = d * (float)sc1,  mm2 = dmin * (float)m1v;
            for (int l = 0; l < 32; l++)
                dst[oi + j + l]      = d1 * (float)(qs[qi+l] & 0x0F) - mm1;
            for (int l = 0; l < 32; l++)
                dst[oi + j + 32 + l] = d2 * (float)(qs[qi+l] >> 4)   - mm2;
            qi += 32; is += 2;
        }
    }
}

static void deq_q6_k(const uint8_t* src, float* dst, uint64_t n) {
    uint64_t nb = n / 256;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t* b  = src + i * 210;
        const uint8_t* ql = b;
        const uint8_t* qh = b + 128;
        const int8_t*  sc = (const int8_t*)(b + 192);
        float d = wtf_f16_to_f32((uint16_t)(b[208] | (b[209] << 8)));
        for (int n_ = 0; n_ < 256; n_ += 128) {
            for (int l = 0; l < 32; l++) {
                int is_ = n_/128*2;
                uint8_t q0 = ql[n_/2+l] & 0xF, q1 = ql[n_/2+l] >> 4;
                uint8_t q2 = ql[n_/2+l+32] & 0xF, q3 = ql[n_/2+l+32] >> 4;
                uint8_t h0 = (qh[n_/4+l] >> 0) & 3, h1 = (qh[n_/4+l] >> 2) & 3;
                uint8_t h2 = (qh[n_/4+l] >> 4) & 3, h3 = (qh[n_/4+l] >> 6) & 3;
                dst[i*256 + n_ + l]      = d * (float)sc[is_+0] * (float)((int)(q0 | (h0<<4)) - 32);
                dst[i*256 + n_ + l + 32] = d * (float)sc[is_+1] * (float)((int)(q1 | (h1<<4)) - 32);
                dst[i*256 + n_ + l + 64] = d * (float)sc[is_+2] * (float)((int)(q2 | (h2<<4)) - 32);
                dst[i*256 + n_ + l + 96] = d * (float)sc[is_+3] * (float)((int)(q3 | (h3<<4)) - 32);
            }
        }
    }
}

static void deq_q5_0(const uint8_t* src, float* dst, uint64_t n) {
    uint64_t nb = n / 32;
    for (uint64_t i = 0; i < nb; i++) {
        const uint8_t* b = src + i * 22;
        float d = wtf_f16_to_f32((uint16_t)(b[0] | (b[1] << 8)));
        uint32_t qh = (uint32_t)b[2] | ((uint32_t)b[3] << 8) |
                      ((uint32_t)b[4] << 16) | ((uint32_t)b[5] << 24);
        const uint8_t* qs = b + 6;
        for (int j = 0; j < 16; j++) {
            int lo = qs[j] & 0x0F, hi = qs[j] >> 4;
            int hb0 = (qh >> j) & 1, hb1 = (qh >> (j+16)) & 1;
            dst[i*32 + j]      = (float)((lo | (hb0<<4)) - 16) * d;
            dst[i*32 + j + 16] = (float)((hi | (hb1<<4)) - 16) * d;
        }
    }
}

// ── Public dispatchers ──────────────────────────────────────────────────────

int wtf_dequant_to_f32(const uint8_t* src, int dtype, uint64_t n, float* dst) {
    switch (dtype) {
    case WTF_DTYPE_F32:  memcpy(dst, src, n * sizeof(float)); return 0;
    case WTF_DTYPE_F16:  deq_f16 (src, dst, n);  return 0;
    case WTF_DTYPE_Q4_0: deq_q4_0(src, dst, n);  return 0;
    case WTF_DTYPE_Q5_0: deq_q5_0(src, dst, n);  return 0;
    case WTF_DTYPE_Q8_0: deq_q8_0(src, dst, n);  return 0;
    case WTF_DTYPE_Q4_K: deq_q4_k(src, dst, n);  return 0;
    case WTF_DTYPE_Q6_K: deq_q6_k(src, dst, n);  return 0;
    default:
        fprintf(stderr, "wtf_kernels: unsupported dtype %d\n", dtype);
        return -1;
    }
}

void wtf_sgemv(float* out, const float* W, const float* x, int m, int n) {
    nt_blas_matvec(out, W, x, m, n);
}

#ifdef USE_BLAS
  #ifdef ACCELERATE
    #include <Accelerate/Accelerate.h>
  #else
    #include <cblas.h>
  #endif
#endif

void wtf_sgemv_strided(float* out, const float* W, int lda,
                       const float* x, int m, int n, int trans) {
#ifdef USE_BLAS
    enum CBLAS_TRANSPOSE t = trans ? CblasTrans : CblasNoTrans;
    cblas_sgemv(CblasRowMajor, t, m, n, 1.0f, W, lda, x, 1, 0.0f, out, 1);
#else
    if (!trans) {
        for (int i = 0; i < m; i++) {
            float s = 0;
            for (int j = 0; j < n; j++) s += W[i * lda + j] * x[j];
            out[i] = s;
        }
    } else {
        for (int j = 0; j < n; j++) out[j] = 0;
        for (int i = 0; i < m; i++) {
            float xi = x[i];
            for (int j = 0; j < n; j++) out[j] += W[i * lda + j] * xi;
        }
    }
#endif
}
