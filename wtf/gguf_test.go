package wtf

import (
	"strings"
	"testing"
)

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

func TestParseMetadataUsesQwenHeadLengths(t *testing.T) {
	meta := parseMetadata(map[string]interface{}{
		"general.architecture":                   "qwen3",
		"qwen3.block_count":                      uint32(28),
		"qwen3.embedding_length":                 uint32(1024),
		"qwen3.attention.head_count":             uint32(16),
		"qwen3.attention.head_count_kv":          uint32(8),
		"qwen3.attention.key_length":             uint32(128),
		"qwen3.attention.value_length":           uint32(128),
		"qwen3.feed_forward_length":              uint32(3072),
		"qwen3.context_length":                   uint32(32768),
		"qwen3.attention.layer_norm_rms_epsilon": float32(1e-6),
		"tokenizer.ggml.tokens":                  []interface{}{"<s>", "</s>"},
		"tokenizer.ggml.bos_token_id":            uint32(0),
		"tokenizer.ggml.eos_token_id":            uint32(1),
	})

	if meta.Architecture != "qwen3" {
		t.Fatalf("Architecture = %q, want qwen3", meta.Architecture)
	}
	if meta.HeadDim != 128 {
		t.Fatalf("HeadDim = %d, want 128", meta.HeadDim)
	}
	if got := meta.NumHeads * meta.HeadDim; got != 2048 {
		t.Fatalf("attention projection dim = %d, want 2048", got)
	}
}

func TestValidateMatrixTensorCatchesWrongShape(t *testing.T) {
	info := &GGUFTensorInfo{
		NDims: 2,
		Dims:  [4]uint64{1024, 1024},
	}
	err := validateMatrixTensor(info, "blk.0.attn_q.weight", 2048, 1024)
	if err == nil {
		t.Fatal("expected shape mismatch")
	}
	if !strings.Contains(err.Error(), "expected [k=1024,m=2048]") {
		t.Fatalf("unexpected error: %v", err)
	}
}
