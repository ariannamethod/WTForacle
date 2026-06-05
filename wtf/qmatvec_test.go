package wtf

// qmatvec_test.go — the packed nt_qmatvec path must agree with the established
// dequant->sgemv path. This is the correctness gate before the Forward hot path
// is switched onto packed weights.

import (
	"math"
	"math/rand"
	"testing"
)

func TestQmatvecQ4_0(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	m, k := 256, 1024 // k % 32 == 0
	nb := k / 32
	wq := make([]byte, m*nb*18)
	for i := range wq {
		wq[i] = byte(rng.Intn(256))
	}
	// sane normal f16 scale (0x2A66) per 18-byte block — avoid inf/nan
	for row := 0; row < m; row++ {
		for b := 0; b < nb; b++ {
			off := (row*nb + b) * 18
			wq[off], wq[off+1] = 0x66, 0x2A
		}
	}
	x := make([]float32, k)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	// oracle: dequant the whole matrix to f32, then the existing sgemv path
	wf, err := dequantToF32(wq, dtypeQ4_0, m*k)
	if err != nil {
		t.Fatalf("dequantToF32: %v", err)
	}
	ref := make([]float32, m)
	sgemv(ref, wf, x, m, k)

	// packed path under test
	got := make([]float32, m)
	if !qmatvec(got, wq, dtypeQ4_0, x, m, k) {
		t.Fatal("qmatvec returned false (unsupported dtype)")
	}

	var maxAbs, maxRef float64
	for i := 0; i < m; i++ {
		if d := math.Abs(float64(ref[i] - got[i])); d > maxAbs {
			maxAbs = d
		}
		if a := math.Abs(float64(ref[i])); a > maxRef {
			maxRef = a
		}
	}
	rel := maxAbs / maxRef
	t.Logf("qmatvec Q4_0 [m=%d k=%d] maxAbs=%.3g maxRef=%.3g rel=%.2g", m, k, maxAbs, maxRef, rel)
	if rel > 1e-3 {
		t.Fatalf("qmatvec Q4_0 diverges from dequant->sgemv: rel=%.3g", rel)
	}
}
