package wtf

import "testing"

func TestGPT2EncodeUsesByteUnicodeAlphabet(t *testing.T) {
	tok := NewTokenizer(&GGUFMetadata{
		TokenModel: "gpt2",
		TokenList: []string{
			"h",
			"i",
			"hi",
			"Ġ",
			"Ġhi",
		},
		TokenMerges: []string{
			"h i",
			"Ġ hi",
		},
		VocabSize: 5,
		BosID:     -1,
		EosID:     -1,
	})

	got := tok.Encode(" hi", false)
	if len(got) != 1 || got[0] != 4 {
		t.Fatalf("Encode(%q) = %v, want [4] for Ġhi", " hi", got)
	}

	if decoded := tok.Decode(got); decoded != " hi" {
		t.Fatalf("Decode(%v) = %q, want %q", got, decoded, " hi")
	}
}
