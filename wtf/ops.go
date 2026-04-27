package wtf

// ops.go — small Go-side activations / norms used by the forward pass.
// Heavy linear algebra (matvec, dequant) lives in notorch.go via cgo.

import "math"

// RMSNorm applies RMS normalization in-place.
func RMSNorm(x []float32, w []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * inv * w[i]
	}
}

// RMSNormInto: out = norm(x) * w. Caller pre-allocates out with len(x).
func RMSNormInto(out, x, w []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		out[i] = x[i] * inv * w[i]
	}
}

// Softmax computes softmax in-place over x[0:n].
func Softmax(x []float32, n int) {
	maxv := x[0]
	for i := 1; i < n; i++ {
		if x[i] > maxv {
			maxv = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - maxv)))
		sum += x[i]
	}
	inv := float32(1.0) / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// SiLU activation: x * sigmoid(x).
func SiLU(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}
