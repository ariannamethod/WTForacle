package main

// tokenizer.go — SentencePiece BPE tokenizer from GGUF metadata
//
// SentencePiece BPE (SmolLM2 / LLaMA) — ▁ prefix, score-based merges
//
// Mode is detected from tokenizer.ggml.model in GGUF metadata.
//
// Token types:
//   1 = normal
//   2 = unknown (<unk>)
//   3 = control (<s>, </s>, etc.)
//   6 = byte fallback (<0x00>...<0xFF>)

import (
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"
)

// gpt2ByteToUnicode builds the GPT-2 byte↔unicode mapping.
// GPT-2 BPE uses a reversible mapping from bytes to printable unicode chars.
// Bytes that are already printable (33-126, 161-172, 174-255) map to themselves.
// Other bytes (0-32, 127-160, 173) map to 256+n.
var gpt2UnicodeToByteMap map[rune]byte

func init() {
	gpt2UnicodeToByteMap = make(map[rune]byte, 256)
	// Printable byte ranges that map to themselves
	var bs []int
	for b := 33; b <= 126; b++ {
		bs = append(bs, b)
	}
	for b := 161; b <= 172; b++ {
		bs = append(bs, b)
	}
	for b := 174; b <= 255; b++ {
		bs = append(bs, b)
	}
	printable := make(map[int]bool, len(bs))
	for _, b := range bs {
		printable[b] = true
	}
	// Identity mappings
	for _, b := range bs {
		gpt2UnicodeToByteMap[rune(b)] = byte(b)
	}
	// Shifted mappings for non-printable bytes
	n := 0
	for b := 0; b < 256; b++ {
		if !printable[b] {
			gpt2UnicodeToByteMap[rune(256+n)] = byte(b)
			n++
		}
	}
}

// Tokenizer handles SentencePiece BPE encoding/decoding
type Tokenizer struct {
	Vocab          []string
	Scores         []float32
	Types          []int32
	VocabSize      int
	BosID          int
	EosID          int
	AddSpacePrefix bool
	IsGPT2         bool // GPT-2 BPE (merge-based) vs SentencePiece (score-based)

	// Lookup table for encoding
	tokenToID map[string]int
	// Byte fallback tokens (SentencePiece style <0xNN>)
	byteTokens [256]int

	// GPT-2 BPE merge priority (pair → rank, lower = higher priority)
	mergePriority map[string]int

	// Special tokens that should be matched as whole units (not BPE'd)
	specialTokens map[string]int
}

// NewTokenizer creates a tokenizer from GGUF metadata
func NewTokenizer(meta *GGUFMetadata) *Tokenizer {
	t := &Tokenizer{
		Vocab:          meta.TokenList,
		Scores:         meta.TokenScores,
		Types:          meta.TokenTypes,
		VocabSize:      meta.VocabSize,
		BosID:          meta.BosID,
		EosID:          meta.EosID,
		AddSpacePrefix: meta.AddSpacePrefix,
	}

	// Build lookup table
	t.tokenToID = make(map[string]int, t.VocabSize)
	for i, tok := range t.Vocab {
		t.tokenToID[tok] = i
	}

	// Map byte fallback tokens (SentencePiece style)
	for i := 0; i < 256; i++ {
		name := fmt.Sprintf("<0x%02X>", i)
		if id, ok := t.tokenToID[name]; ok {
			t.byteTokens[i] = id
		} else {
			t.byteTokens[i] = -1
		}
	}

	// Build special tokens map (control tokens that should not be BPE'd)
	t.specialTokens = make(map[string]int)
	if t.Types != nil {
		for i, typ := range t.Types {
			if typ == 3 && i < len(t.Vocab) { // type 3 = control token
				token := t.Vocab[i]
				if len(token) > 2 { // skip empty/single char control tokens
					t.specialTokens[token] = i
				}
			}
		}
		fmt.Printf("[tongue/tokenizer] %d special tokens registered\n", len(t.specialTokens))
	}

	// GPT-2 BPE: build merge priority map
	if meta.TokenModel == "gpt2" || (len(meta.TokenMerges) > 0 && len(meta.TokenScores) == 0) {
		t.IsGPT2 = true
		t.AddSpacePrefix = false // GPT-2 doesn't use space prefix
		t.mergePriority = make(map[string]int, len(meta.TokenMerges))
		for i, merge := range meta.TokenMerges {
			t.mergePriority[merge] = i
		}
		fmt.Printf("[tongue/tokenizer] GPT-2 BPE mode: %d merges loaded\n", len(t.mergePriority))
	}

	fmt.Printf("[tongue/tokenizer] vocab=%d bos=%d eos=%d add_space_prefix=%v\n",
		t.VocabSize, t.BosID, t.EosID, t.AddSpacePrefix)
	return t
}

// Encode converts text to token IDs using BPE
func (t *Tokenizer) Encode(text string, addBos bool) []int {
	var tokens []int

	if addBos && t.BosID >= 0 {
		tokens = append(tokens, t.BosID)
	}

	if len(text) == 0 {
		return tokens
	}

	// Split text on special tokens, encode each segment
	segments := t.splitOnSpecialTokens(text)
	for _, seg := range segments {
		if id, ok := t.specialTokens[seg]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.encodeSentencePiece(seg)...)
		}
	}

	return tokens
}

// splitOnSpecialTokens splits text into segments, preserving special tokens as separate items
func (t *Tokenizer) splitOnSpecialTokens(text string) []string {
	if len(t.specialTokens) == 0 {
		return []string{text}
	}

	var segments []string
	remaining := text

	for len(remaining) > 0 {
		// Find earliest special token in remaining text
		bestPos := -1
		bestLen := 0
		bestToken := ""

		for token := range t.specialTokens {
			pos := strings.Index(remaining, token)
			if pos >= 0 && (bestPos < 0 || pos < bestPos || (pos == bestPos && len(token) > bestLen)) {
				bestPos = pos
				bestLen = len(token)
				bestToken = token
			}
		}

		if bestPos < 0 {
			// No more special tokens found
			if len(remaining) > 0 {
				segments = append(segments, remaining)
			}
			break
		}

		// Add text before special token
		if bestPos > 0 {
			segments = append(segments, remaining[:bestPos])
		}
		// Add special token
		segments = append(segments, bestToken)
		remaining = remaining[bestPos+bestLen:]
	}

	return segments
}

// encodeSentencePiece does BPE encoding (SentencePiece or GPT-2)
func (t *Tokenizer) encodeSentencePiece(text string) []int {
	if t.IsGPT2 {
		return t.encodeGPT2(text)
	}

	// SentencePiece: prepend space if configured
	if t.AddSpacePrefix && len(text) > 0 && text[0] != ' ' {
		text = " " + text
	}

	// SentencePiece replaces spaces with ▁ (U+2581)
	text = strings.ReplaceAll(text, " ", "▁")

	// Initial tokenization: split into individual characters/codepoints
	symbols := t.initialTokenizeSP(text)

	// BPE merge loop: find highest-score merge
	symbols = t.bpeMerge(symbols)

	// Convert symbols to token IDs
	return t.symbolsToIDs(symbols)
}

// encodeGPT2 does GPT-2 BPE encoding (byte-level, merge-based)
func (t *Tokenizer) encodeGPT2(text string) []int {
	// GPT-2 BPE: each byte is an initial symbol (using vocab tokens)
	var symbols []string
	for _, b := range []byte(text) {
		// Try the byte as a single-char string first
		ch := string([]byte{b})
		if _, ok := t.tokenToID[ch]; ok {
			symbols = append(symbols, ch)
		} else {
			// Byte fallback
			byteStr := fmt.Sprintf("<0x%02X>", b)
			symbols = append(symbols, byteStr)
		}
	}

	symbols = t.bpeMerge(symbols)
	return t.symbolsToIDs(symbols)
}

// bpeMerge applies greedy BPE merging.
// GPT-2 mode: uses merge priority (pair → rank, lower = merge first).
// SentencePiece mode: uses token scores (higher score = merge first).
func (t *Tokenizer) bpeMerge(symbols []string) []string {
	if t.IsGPT2 {
		return t.bpeMergeGPT2(symbols)
	}
	return t.bpeMergeScores(symbols)
}

// bpeMergeGPT2 uses merge priority table (GPT-2 / SmolLM2 style)
func (t *Tokenizer) bpeMergeGPT2(symbols []string) []string {
	for {
		bestRank := len(t.mergePriority) + 1 // worse than any real rank
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.mergePriority[pair]; ok {
				if rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

// bpeMergeScores uses token scores (SentencePiece / LLaMA style)
func (t *Tokenizer) bpeMergeScores(symbols []string) []string {
	for {
		bestScore := float32(-1e30)
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			merged := symbols[i] + symbols[i+1]
			if id, ok := t.tokenToID[merged]; ok && id < len(t.Scores) {
				score := t.Scores[id]
				if score > bestScore {
					bestScore = score
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

// symbolsToIDs converts BPE symbols to token IDs with byte fallback
func (t *Tokenizer) symbolsToIDs(symbols []string) []int {
	var tokens []int
	for _, sym := range symbols {
		if id, ok := t.tokenToID[sym]; ok {
			tokens = append(tokens, id)
		} else {
			// Fall back to byte tokens
			for _, b := range []byte(sym) {
				if t.byteTokens[b] >= 0 {
					tokens = append(tokens, t.byteTokens[b])
				}
			}
		}
	}
	return tokens
}

// initialTokenizeSP splits text into initial symbols for SentencePiece BPE
func (t *Tokenizer) initialTokenizeSP(text string) []string {
	var symbols []string

	runes := []rune(text)
	i := 0
	for i < len(runes) {
		// Try single character in vocab
		ch := string(runes[i])
		if _, ok := t.tokenToID[ch]; ok {
			symbols = append(symbols, ch)
			i++
			continue
		}

		// Fall back to byte representation
		for _, b := range []byte(string(runes[i])) {
			byteStr := fmt.Sprintf("<0x%02X>", b)
			symbols = append(symbols, byteStr)
		}
		i++
	}

	return symbols
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= t.VocabSize {
			continue
		}
		piece := t.Vocab[id]

		// Skip control tokens
		if t.Types != nil && id < len(t.Types) && t.Types[id] == 3 {
			continue
		}

		// Handle byte fallback tokens (<0xNN>)
		if len(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' {
			var b byte
			fmt.Sscanf(piece, "<0x%02X>", &b)
			sb.WriteByte(b)
			continue
		}

		if t.IsGPT2 {
			// GPT-2: decode unicode chars back to bytes
			sb.WriteString(gpt2DecodePiece(piece))
		} else {
			// SentencePiece: ▁ -> space
			piece = strings.ReplaceAll(piece, "▁", " ")
			sb.WriteString(piece)
		}
	}

	result := sb.String()
	// Trim leading space for SentencePiece
	if t.AddSpacePrefix && len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}
	return result
}

// gpt2DecodePiece reverses GPT-2 byte encoding for a token piece
func gpt2DecodePiece(piece string) string {
	var buf []byte
	for len(piece) > 0 {
		r, size := utf8.DecodeRuneInString(piece)
		if b, ok := gpt2UnicodeToByteMap[r]; ok {
			buf = append(buf, b)
		} else {
			// Unknown rune — pass through as UTF-8
			buf = append(buf, piece[:size]...)
		}
		piece = piece[size:]
	}
	return string(buf)
}

// DecodeToken decodes a single token ID
func (t *Tokenizer) DecodeToken(id int) string {
	if id < 0 || id >= t.VocabSize {
		return ""
	}
	piece := t.Vocab[id]

	// Handle byte fallback
	if len(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' {
		var b byte
		fmt.Sscanf(piece, "<0x%02X>", &b)
		return string([]byte{b})
	}

	if t.IsGPT2 {
		return gpt2DecodePiece(piece)
	}

	// SentencePiece: ▁ -> space
	piece = strings.ReplaceAll(piece, "▁", " ")
	return piece
}

// FindSpecialToken searches for a special token by name
func (t *Tokenizer) FindSpecialToken(name string) int {
	variants := []string{
		name,
		"<|" + name + "|>",
		"<" + name + ">",
	}
	for _, v := range variants {
		if id, ok := t.tokenToID[v]; ok {
			return id
		}
	}
	return -1
}

// DebugTokenize shows tokens for debugging
func (t *Tokenizer) DebugTokenize(text string) {
	ids := t.Encode(text, false)
	fmt.Printf("[tokenizer] '%s' -> %d tokens: ", text, len(ids))
	for _, id := range ids {
		if id >= 0 && id < t.VocabSize {
			fmt.Printf("[%d:'%s'] ", id, t.Vocab[id])
		}
	}
	fmt.Println()
}

// SortVocabByScore returns vocab indices sorted by score (for debug)
func (t *Tokenizer) SortVocabByScore() []int {
	idx := make([]int, t.VocabSize)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return t.Scores[idx[i]] > t.Scores[idx[j]]
	})
	return idx
}
