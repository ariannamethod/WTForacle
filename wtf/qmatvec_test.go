package wtf

// qmatvec_test.go — the packed nt_qmatvec path must agree with the established
// dequant->sgemv path. This is the correctness gate before the Forward hot path
// is switched onto packed weights.

import (
	"math"
	"math/rand"
	"testing"
)

func TestQmatvecPackedDtypes(t *testing.T) {
	tests := []struct {
		name       string
		dtype      int
		k          int
		blockBytes int
		initBlock  func([]byte, *rand.Rand)
	}{
		{name: "q4_0", dtype: dtypeQ4_0, k: 1024, blockBytes: 18, initBlock: initQ4_0Block},
		{name: "q5_0", dtype: dtypeQ5_0, k: 1024, blockBytes: 22, initBlock: initQ5_0Block},
		{name: "q8_0", dtype: dtypeQ8_0, k: 1024, blockBytes: 34, initBlock: initQ8_0Block},
		{name: "q4_k", dtype: dtypeQ4_K, k: 1024, blockBytes: 144, initBlock: initQ4KBlock},
		{name: "q6_k", dtype: dtypeQ6_K, k: 1024, blockBytes: 210, initBlock: initQ6KBlock},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(42))
			m := 128
			blockElems := ggmlBlockElements(uint32(tt.dtype))
			nb := tt.k / blockElems
			wq := make([]byte, m*nb*tt.blockBytes)
			for row := 0; row < m; row++ {
				for b := 0; b < nb; b++ {
					off := (row*nb + b) * tt.blockBytes
					tt.initBlock(wq[off:off+tt.blockBytes], rng)
				}
			}
			x := make([]float32, tt.k)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}

			wf, err := dequantToF32(wq, uint32(tt.dtype), m*tt.k)
			if err != nil {
				t.Fatalf("dequantToF32: %v", err)
			}
			ref := make([]float32, m)
			sgemv(ref, wf, x, m, tt.k)

			got := make([]float32, m)
			if !qmatvec(got, wq, tt.dtype, x, m, tt.k) {
				t.Fatal("qmatvec returned false (unsupported dtype)")
			}

			maxAbs, maxRef := maxDiff(ref, got)
			rel := maxAbs / maxRef
			t.Logf("qmatvec %s [m=%d k=%d] maxAbs=%.3g maxRef=%.3g rel=%.2g",
				tt.name, m, tt.k, maxAbs, maxRef, rel)
			if rel > 1e-3 {
				t.Fatalf("qmatvec %s diverges from dequant->sgemv: rel=%.3g", tt.name, rel)
			}
		})
	}
}

func TestQmatvecI8ApproxDtypes(t *testing.T) {
	tests := []struct {
		name       string
		dtype      int
		k          int
		blockBytes int
		initBlock  func([]byte, *rand.Rand)
	}{
		{name: "q4_0", dtype: dtypeQ4_0, k: 1024, blockBytes: 18, initBlock: initQ4_0Block},
		{name: "q8_0", dtype: dtypeQ8_0, k: 1024, blockBytes: 34, initBlock: initQ8_0Block},
		{name: "q6_k", dtype: dtypeQ6_K, k: 1024, blockBytes: 210, initBlock: initQ6KBlock},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(84))
			m := 128
			blockElems := ggmlBlockElements(uint32(tt.dtype))
			nb := tt.k / blockElems
			wq := make([]byte, m*nb*tt.blockBytes)
			for row := 0; row < m; row++ {
				for b := 0; b < nb; b++ {
					off := (row*nb + b) * tt.blockBytes
					tt.initBlock(wq[off:off+tt.blockBytes], rng)
				}
			}
			x := make([]float32, tt.k)
			for i := range x {
				x[i] = rng.Float32()*2 - 1
			}

			ref := make([]float32, m)
			if !qmatvec(ref, wq, tt.dtype, x, m, tt.k) {
				t.Fatal("qmatvec returned false (unsupported dtype)")
			}
			got := make([]float32, m)
			if !qmatvecI8(got, wq, tt.dtype, x, m, tt.k) {
				t.Fatal("qmatvecI8 returned false (unsupported dtype)")
			}

			maxAbs, maxRef := maxDiff(ref, got)
			rel := maxAbs / maxRef
			t.Logf("qmatvec_i8 %s [m=%d k=%d] maxAbs=%.3g maxRef=%.3g rel=%.2g",
				tt.name, m, tt.k, maxAbs, maxRef, rel)
			if rel > 2e-2 {
				t.Fatalf("qmatvec_i8 %s diverges too far from exact qmatvec: rel=%.3g", tt.name, rel)
			}
		})
	}
}

func TestQmatvecI8UnsupportedDtypes(t *testing.T) {
	rng := rand.New(rand.NewSource(126))
	tests := []struct {
		name       string
		dtype      int
		k          int
		blockBytes int
		initBlock  func([]byte, *rand.Rand)
	}{
		{name: "q5_0", dtype: dtypeQ5_0, k: 1024, blockBytes: 22, initBlock: initQ5_0Block},
		{name: "q4_k", dtype: dtypeQ4_K, k: 1024, blockBytes: 144, initBlock: initQ4KBlock},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			m := 4
			blockElems := ggmlBlockElements(uint32(tt.dtype))
			nb := tt.k / blockElems
			wq := make([]byte, m*nb*tt.blockBytes)
			for row := 0; row < m; row++ {
				for b := 0; b < nb; b++ {
					off := (row*nb + b) * tt.blockBytes
					tt.initBlock(wq[off:off+tt.blockBytes], rng)
				}
			}
			x := make([]float32, tt.k)
			out := make([]float32, m)
			if qmatvecI8(out, wq, tt.dtype, x, m, tt.k) {
				t.Fatal("qmatvecI8 accepted unsupported dtype")
			}
		})
	}
}

func initQ4_0Block(b []byte, rng *rand.Rand) {
	fillRandom(b, rng)
	writeHalfScale(b, 0)
}

func initQ5_0Block(b []byte, rng *rand.Rand) {
	fillRandom(b, rng)
	writeHalfScale(b, 0)
}

func initQ8_0Block(b []byte, rng *rand.Rand) {
	fillRandom(b, rng)
	writeHalfScale(b, 0)
}

func initQ4KBlock(b []byte, rng *rand.Rand) {
	fillRandom(b, rng)
	writeHalfScale(b, 0)
	writeHalfScale(b, 2)
	for i := 4; i < 16; i++ {
		b[i] = byte(1 + rng.Intn(15))
	}
}

func initQ6KBlock(b []byte, rng *rand.Rand) {
	fillRandom(b, rng)
	for i := 192; i < 208; i++ {
		b[i] = byte(int8(rng.Intn(7) - 3))
	}
	writeHalfScale(b, 208)
}

func fillRandom(b []byte, rng *rand.Rand) {
	for i := range b {
		b[i] = byte(rng.Intn(256))
	}
}

func writeHalfScale(b []byte, off int) {
	// 0x2A66 is a small finite fp16 scale; random scale bits can encode NaN/Inf.
	b[off], b[off+1] = 0x66, 0x2A
}

func maxDiff(ref, got []float32) (maxAbs, maxRef float64) {
	for i := range ref {
		if d := math.Abs(float64(ref[i] - got[i])); d > maxAbs {
			maxAbs = d
		}
		if a := math.Abs(float64(ref[i])); a > maxRef {
			maxRef = a
		}
	}
	if maxRef == 0 {
		maxRef = 1
	}
	return maxAbs, maxRef
}
