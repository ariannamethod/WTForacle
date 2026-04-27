package wtf

// notorch.go — cgo bindings for the vendored notorch + gguf + wtf_kernels
// stack in ../ariannamethod/. Two primitives reach Go: Dequant (any GGML
// quant tag → contiguous float32) and SGEMV (out = W @ x).
//
// Building requires cgo + a BLAS provider:
//   macOS — Apple Accelerate (zero deps, AMX path)
//   Linux — OpenBLAS (apt install libopenblas-dev)
//
// notorch.c is built with USE_BLAS so nt_blas_matvec routes to cblas_sgemv.

/*
#cgo CFLAGS: -O3 -DUSE_BLAS -I${SRCDIR}/../ariannamethod
#cgo darwin CFLAGS: -DACCELERATE -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lopenblas

#include <stdint.h>
#include "wtf_kernels.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// GGML tensor type tags (mirror gguf.h / wtf_kernels.h).
const (
	dtypeF32  = 0
	dtypeF16  = 1
	dtypeQ4_0 = 2
	dtypeQ5_0 = 6
	dtypeQ8_0 = 8
	dtypeQ4_K = 12
	dtypeQ6_K = 14
)

// dequantToF32 unpacks `n` elements of the given GGML dtype from src into a
// freshly allocated []float32. Routes through the vendored notorch kernel.
func dequantToF32(src []byte, dtype uint32, n int) ([]float32, error) {
	if n <= 0 {
		return nil, fmt.Errorf("dequantToF32: n=%d", n)
	}
	dst := make([]float32, n)
	rc := C.wtf_dequant_to_f32(
		(*C.uint8_t)(unsafe.Pointer(&src[0])),
		C.int(dtype),
		C.uint64_t(n),
		(*C.float)(unsafe.Pointer(&dst[0])),
	)
	if rc != 0 {
		return nil, fmt.Errorf("dequantToF32: unsupported dtype %d", dtype)
	}
	return dst, nil
}

// sgemv computes out[m] = W[m,n] @ x[n] via cblas_sgemv (Accelerate / OpenBLAS).
func sgemv(out, w, x []float32, m, n int) {
	C.wtf_sgemv(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(m), C.int(n),
	)
}

// sgemvStrided is sgemv against a sub-matrix view (row stride lda > n).
// trans=false: out[m]      = W[m,n] @ x[n]
// trans=true:  out[n]      = W[m,n]^T @ x[m]
// Used for the KV-cache attention loops where each head reads from a [pos+1,
// head_dim] window inside a [seq_len, kv_dim] buffer.
func sgemvStrided(out, w []float32, lda int, x []float32, m, n int, trans bool) {
	t := C.int(0)
	if trans {
		t = 1
	}
	C.wtf_sgemv_strided(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.float)(unsafe.Pointer(&w[0])),
		C.int(lda),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(m), C.int(n), t,
	)
}
