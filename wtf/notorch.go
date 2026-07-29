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
#cgo CFLAGS: -O3 -DUSE_BLAS -pthread -I${SRCDIR}/../ariannamethod
#cgo darwin CFLAGS: -DACCELERATE -DACCELERATE_NEW_LAPACK -Wno-deprecated-declarations
#cgo darwin LDFLAGS: -framework Accelerate
#cgo linux LDFLAGS: -lopenblas

#include <stdint.h>
#include "wtf_kernels.h"
*/
import "C"

import (
	"fmt"
	"os"
	"strings"
	"unsafe"
)

var useQmatvecI8 = envFlag("WTF_QMATVEC_I8")

func envFlag(name string) bool {
	v := strings.ToLower(strings.TrimSpace(os.Getenv(name)))
	return v != "" && v != "0" && v != "false" && v != "off" && v != "no"
}

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
	if err := dequantToF32Into(src, dtype, dst, n); err != nil {
		return nil, err
	}
	return dst, nil
}

// dequantToF32Into unpacks into an existing destination buffer. It is used by
// packed embedding row lookup, where allocating a new []float32 per token would
// erase the memory win.
func dequantToF32Into(src []byte, dtype uint32, dst []float32, n int) error {
	if n <= 0 {
		return fmt.Errorf("dequantToF32Into: n=%d", n)
	}
	if len(dst) < n {
		return fmt.Errorf("dequantToF32Into: dst len=%d, need %d", len(dst), n)
	}
	if len(src) == 0 {
		return fmt.Errorf("dequantToF32Into: empty source")
	}
	rc := C.wtf_dequant_to_f32(
		(*C.uint8_t)(unsafe.Pointer(&src[0])),
		C.int(dtype),
		C.uint64_t(n),
		(*C.float)(unsafe.Pointer(&dst[0])),
	)
	if rc != 0 {
		return fmt.Errorf("dequantToF32: unsupported dtype %d", dtype)
	}
	return nil
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

// qmatvec computes out[m] = Wq[m,k] @ x[k] from PACKED quantized weights (raw
// GGUF bytes, dtype = GGML tag), dequantized inline by notorch's nt_qmatvec —
// no dense-f32 blow-up. Returns false if the dtype has no packed kernel.
func qmatvec(out []float32, wq []byte, dtype int, x []float32, m, k int) bool {
	rc := C.wtf_qmatvec(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.uint8_t)(unsafe.Pointer(&wq[0])),
		C.int(dtype),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(m), C.int(k),
	)
	return rc == 0
}

// qmatvecI8 is the approximate fast path: x is quantized to int8 per call and
// multiplied against packed weights. It is intentionally separate from qmatvec
// so generation can opt into it only after quality gates pass.
func qmatvecI8(out []float32, wq []byte, dtype int, x []float32, m, k int) bool {
	rc := C.wtf_qmatvec_i8(
		(*C.float)(unsafe.Pointer(&out[0])),
		(*C.uint8_t)(unsafe.Pointer(&wq[0])),
		C.int(dtype),
		(*C.float)(unsafe.Pointer(&x[0])),
		C.int(m), C.int(k),
	)
	return rc == 0
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
