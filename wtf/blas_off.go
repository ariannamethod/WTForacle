//go:build !blas

package main

// Stub: no BLAS acceleration. Pure Go fallback.
// Build with -tags blas to enable hardware acceleration.

var useBLAS = false

func blasMatMulF32(out []float32, w []float32, x []float32, rows, cols int) {}
func blasDot(a, b []float32) float32 { return 0 }
