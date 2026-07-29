package wtf

import (
	"math/rand"
	"testing"
)

func TestQTableLookupPackedMatchesFullDequant(t *testing.T) {
	tests := []struct {
		name       string
		dtype      int
		cols       int
		blockBytes int
		initBlock  func([]byte, *rand.Rand)
	}{
		{name: "q4_0", dtype: dtypeQ4_0, cols: 1024, blockBytes: 18, initBlock: initQ4_0Block},
		{name: "q5_0", dtype: dtypeQ5_0, cols: 1024, blockBytes: 22, initBlock: initQ5_0Block},
		{name: "q8_0", dtype: dtypeQ8_0, cols: 1024, blockBytes: 34, initBlock: initQ8_0Block},
		{name: "q4_k", dtype: dtypeQ4_K, cols: 1024, blockBytes: 144, initBlock: initQ4KBlock},
		{name: "q6_k", dtype: dtypeQ6_K, cols: 1024, blockBytes: 210, initBlock: initQ6KBlock},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rng := rand.New(rand.NewSource(168))
			rows := 7
			blockElems := ggmlBlockElements(uint32(tt.dtype))
			nb := tt.cols / blockElems
			rowBytes, err := qtableRowBytes(uint32(tt.dtype), tt.cols)
			if err != nil {
				t.Fatalf("qtableRowBytes: %v", err)
			}
			if rowBytes != nb*tt.blockBytes {
				t.Fatalf("rowBytes=%d, want %d", rowBytes, nb*tt.blockBytes)
			}

			wq := make([]byte, rows*rowBytes)
			for row := 0; row < rows; row++ {
				for b := 0; b < nb; b++ {
					off := row*rowBytes + b*tt.blockBytes
					tt.initBlock(wq[off:off+tt.blockBytes], rng)
				}
			}

			full, err := dequantToF32(wq, uint32(tt.dtype), rows*tt.cols)
			if err != nil {
				t.Fatalf("dequantToF32: %v", err)
			}
			table := QTable{
				Packed:   wq,
				Dtype:    tt.dtype,
				Rows:     rows,
				Cols:     tt.cols,
				RowBytes: rowBytes,
			}

			for row := 0; row < rows; row++ {
				got := make([]float32, tt.cols)
				table.lookup(got, row)
				maxAbs, maxRef := maxDiff(full[row*tt.cols:(row+1)*tt.cols], got)
				rel := maxAbs / maxRef
				if rel > 1e-6 {
					t.Fatalf("row %d rel=%.3g maxAbs=%.3g maxRef=%.3g", row, rel, maxAbs, maxRef)
				}
			}
		})
	}
}

func TestQTableMatvecPackedMatchesFullDequant(t *testing.T) {
	rng := rand.New(rand.NewSource(336))
	rows, cols := 128, 1024
	dtype := dtypeQ8_0
	rowBytes, err := qtableRowBytes(uint32(dtype), cols)
	if err != nil {
		t.Fatalf("qtableRowBytes: %v", err)
	}
	nb := cols / ggmlBlockElements(uint32(dtype))
	wq := make([]byte, rows*rowBytes)
	for row := 0; row < rows; row++ {
		for b := 0; b < nb; b++ {
			off := row*rowBytes + b*34
			initQ8_0Block(wq[off:off+34], rng)
		}
	}
	x := make([]float32, cols)
	for i := range x {
		x[i] = rng.Float32()*2 - 1
	}

	full, err := dequantToF32(wq, uint32(dtype), rows*cols)
	if err != nil {
		t.Fatalf("dequantToF32: %v", err)
	}
	ref := make([]float32, rows)
	sgemv(ref, full, x, rows, cols)

	got := make([]float32, rows)
	table := QTable{Packed: wq, Dtype: dtype, Rows: rows, Cols: cols, RowBytes: rowBytes}
	table.matvec(got, x)

	maxAbs, maxRef := maxDiff(ref, got)
	rel := maxAbs / maxRef
	if rel > 1e-3 {
		t.Fatalf("packed table matvec diverges: rel=%.3g maxAbs=%.3g maxRef=%.3g", rel, maxAbs, maxRef)
	}
}

func TestQTablePackedSupportRequiresRowAlignedBlocks(t *testing.T) {
	if qtablePackedSupported(dtypeQ4_K, 960) {
		t.Fatal("Q4_K table with 960 columns should fall back: one row is not 256-aligned")
	}
	if !qtablePackedSupported(dtypeQ4_K, 1024) {
		t.Fatal("Q4_K table with 1024 columns should be packable")
	}
	if !packedMatvecSupported(dtypeQ8_0, 960) {
		t.Fatal("Q8_0 matrix with 960 columns should be packable")
	}
}
