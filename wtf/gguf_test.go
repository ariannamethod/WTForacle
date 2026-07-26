package wtf

import "testing"

func TestGGMLBlockMetadata(t *testing.T) {
	tests := []struct {
		name     string
		typ      uint32
		size     int
		elements int
	}{
		{name: "q4_0", typ: ggmlTypeQ4_0, size: 18, elements: 32},
		{name: "q4_1", typ: ggmlTypeQ4_1, size: 20, elements: 32},
		{name: "q5_0", typ: ggmlTypeQ5_0, size: 22, elements: 32},
		{name: "q5_1", typ: ggmlTypeQ5_1, size: 24, elements: 32},
		{name: "q8_0", typ: ggmlTypeQ8_0, size: 34, elements: 32},
		{name: "q4_k", typ: ggmlTypeQ4_K, size: 144, elements: 256},
		{name: "q6_k", typ: ggmlTypeQ6_K, size: 210, elements: 256},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := ggmlBlockSize(tt.typ); got != tt.size {
				t.Fatalf("ggmlBlockSize(%s) = %d, want %d", tt.name, got, tt.size)
			}
			if got := ggmlBlockElements(tt.typ); got != tt.elements {
				t.Fatalf("ggmlBlockElements(%s) = %d, want %d", tt.name, got, tt.elements)
			}
		})
	}
}
