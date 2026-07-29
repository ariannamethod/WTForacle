package wtf

import "testing"

func TestNormalizeMaxSeqLen(t *testing.T) {
	tests := []struct {
		name string
		in   int
		want int
	}{
		{name: "default", in: 0, want: DefaultMaxSeqLen},
		{name: "negative", in: -1, want: DefaultMaxSeqLen},
		{name: "minimum", in: 1, want: minMaxSeqLen},
		{name: "explicit", in: 512, want: 512},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := normalizeMaxSeqLen(tt.in); got != tt.want {
				t.Fatalf("normalizeMaxSeqLen(%d) = %d, want %d", tt.in, got, tt.want)
			}
		})
	}
}
