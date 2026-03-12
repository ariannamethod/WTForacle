package main

// CGO bridge: WTForacle engine (SmolLM2 360M GGUF)
//
// Cynical reddit oracle. 360M parameters fine-tuned on pure snark.
// Anti-emoji, anti-loop, anti-boring.
//
// Build as shared library:
//   go build -buildmode=c-shared -o libwtf.dylib ./cmd/wtf-lib/
//
// C interface: wtf_init, wtf_free, wtf_generate, wtf_encode, wtf_decode_token
//
// Optimized: zero-alloc sampling, pre-allocated buffers, no GC pressure in hot loop.

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	"fmt"
	"sync"
	"unsafe"

	"wtforacle/wtf"
)

// Global state (singleton)
var (
	gModel     *wtf.LlamaModel
	gTokenizer *wtf.Tokenizer
	gGGUF      *wtf.GGUFFile
	gMu        sync.Mutex

	// Temperature floor: never freezes
	tempFloor float32 = 0.9

	// Repetition penalty: matches Arianna's proven settings
	repPenalty float32 = 1.15
	repWindow  int     = 64

	// Frequency penalty: disabled by default (too aggressive = kills English, leaks Chinese)
	freqPenalty float32 = 0.0

	// Pre-allocated sampling buffers (zero-alloc hot path)
	gSampleBuf *wtf.SampleBuffers
)

// ============================================================
// Lifecycle
// ============================================================

//export wtf_init
func wtf_init(weightsPath *C.char) C.int {
	gMu.Lock()
	defer gMu.Unlock()

	path := C.GoString(weightsPath)
	fmt.Printf("[wtf] loading GGUF from %s\n", path)

	type initResult struct {
		gguf      *wtf.GGUFFile
		model     *wtf.LlamaModel
		tokenizer *wtf.Tokenizer
		err       error
	}
	ch := make(chan initResult, 1)
	go func() {
		var r initResult
		r.gguf, r.err = wtf.LoadGGUF(path)
		if r.err != nil {
			ch <- r
			return
		}
		r.model, r.err = wtf.LoadLlamaModel(r.gguf)
		if r.err != nil {
			ch <- r
			return
		}
		r.tokenizer = wtf.NewTokenizer(&r.gguf.Meta)
		ch <- r
	}()
	r := <-ch

	if r.err != nil {
		fmt.Printf("[wtf] ERROR: %v\n", r.err)
		return -1
	}

	gGGUF = r.gguf
	gModel = r.model
	gTokenizer = r.tokenizer

	// Pre-allocate sampling buffers (once, reused every token)
	gSampleBuf = wtf.NewSampleBuffers(gModel.Config.VocabSize)

	fmt.Printf("[wtf] initialized: %d layers, %d dim, %d vocab, temp_floor=%.1f, rep=%.2f, freq=%.2f, window=%d\n",
		gModel.Config.NumLayers, gModel.Config.EmbedDim,
		gModel.Config.VocabSize, tempFloor, repPenalty, freqPenalty, repWindow)

	return 0
}

//export wtf_free
func wtf_free() {
	gMu.Lock()
	defer gMu.Unlock()

	gModel = nil
	gTokenizer = nil
	gGGUF = nil
	gSampleBuf = nil
	fmt.Println("[wtf] freed")
}

// ============================================================
// Settings
// ============================================================

//export wtf_set_temp_floor
func wtf_set_temp_floor(floor C.float) {
	tempFloor = float32(floor)
}

//export wtf_set_rep_penalty
func wtf_set_rep_penalty(penalty C.float, window C.int) {
	repPenalty = float32(penalty)
	repWindow = int(window)
}

//export wtf_set_freq_penalty
func wtf_set_freq_penalty(penalty C.float) {
	freqPenalty = float32(penalty)
}

// ============================================================
// Generation
// ============================================================

//export wtf_reset
func wtf_reset() {
	gMu.Lock()
	defer gMu.Unlock()
	if gModel != nil {
		gModel.Reset()
	}
}

//export wtf_generate
func wtf_generate(
	promptC *C.char,
	outputC *C.char, maxOutputLen C.int,
	maxTokens C.int,
	temperature C.float, topP C.float,
	anchorPromptC *C.char,
) C.int {
	gMu.Lock()
	defer gMu.Unlock()

	if gModel == nil || gTokenizer == nil {
		return 0
	}

	prompt := C.GoString(promptC)
	anchorPrompt := ""
	if anchorPromptC != nil {
		anchorPrompt = C.GoString(anchorPromptC)
	}

	maxTok := int(maxTokens)
	maxOut := int(maxOutputLen) - 1
	temp := float32(temperature)
	if temp < tempFloor {
		temp = tempFloor
	}
	tp := float32(topP)

	// Run generation in goroutine (full Go stack) to avoid cgo stack limits
	type genResult struct {
		output   []byte
		genCount int
	}
	ch := make(chan genResult, 1)
	go func() {
		// Build token sequence: [optional BOS] + raw anchor + raw user tokens
		var allTokens []int
		if gTokenizer.BosID >= 0 && gTokenizer.BosID != gTokenizer.EosID {
			allTokens = append(allTokens, gTokenizer.BosID)
		}
		if anchorPrompt != "" {
			anchorTokens := gTokenizer.Encode(anchorPrompt, false)
			allTokens = append(allTokens, anchorTokens...)
		}
		userTokens := gTokenizer.Encode(prompt, false)
		allTokens = append(allTokens, userTokens...)
		gModel.Reset()

		// Prefill: feed all prompt tokens through transformer
		pos := 0
		for _, tok := range allTokens {
			gModel.Forward(tok, pos)
			pos++
			if pos >= gModel.Config.SeqLen-1 {
				break
			}
		}

		// Generate (zero-alloc hot loop)
		output := make([]byte, 0, 2048)
		genCount := 0
		graceLimit := 32
		inGrace := false
		recentTokens := make([]int, 0, repWindow)
		tokenCounts := make(map[int]int, 64)
		vocab := gModel.Config.VocabSize

		for i := 0; i < maxTok+graceLimit && len(output) < maxOut; i++ {
			if i >= maxTok && !inGrace {
				inGrace = true
			}
			if inGrace && len(output) > 0 {
				last := output[len(output)-1]
				if last == '.' || last == '!' || last == '?' || last == '\n' {
					break
				}
			}

			// Repetition penalty (presence-based)
			if repPenalty > 1.0 {
				for _, tok := range recentTokens {
					logit := gModel.State.Logits[tok]
					if logit > 0 {
						gModel.State.Logits[tok] = logit / repPenalty
					} else {
						gModel.State.Logits[tok] = logit * repPenalty
					}
				}
			}

			// Frequency penalty (count-based)
			if freqPenalty > 0 {
				for tok, count := range tokenCounts {
					gModel.State.Logits[tok] -= freqPenalty * float32(count)
				}
			}

			// Sample next token (zero-alloc)
			var next int
			if tp < 1.0 {
				next = wtf.SampleTopP(gModel.State.Logits, vocab, temp, tp, gSampleBuf)
			} else {
				next = wtf.SampleTopK(gModel.State.Logits, vocab, temp, 50, gSampleBuf)
			}

			// Update frequency counts + sliding window
			tokenCounts[next]++
			recentTokens = append(recentTokens, next)
			if len(recentTokens) > repWindow {
				leaving := recentTokens[0]
				tokenCounts[leaving]--
				if tokenCounts[leaving] <= 0 {
					delete(tokenCounts, leaving)
				}
				recentTokens = recentTokens[1:]
			}

			// Stop on EOS
			if next == gTokenizer.EosID {
				break
			}

			// Cycle detection: last 8 tokens == previous 8 tokens
			if len(recentTokens) >= 16 {
				n := len(recentTokens)
				isCycle := true
				for k := 0; k < 8; k++ {
					if recentTokens[n-1-k] != recentTokens[n-9-k] {
						isCycle = false
						break
					}
				}
				if isCycle {
					fmt.Println("[wtf] cycle detected, breaking")
					break
				}
			}

			piece := gTokenizer.DecodeToken(next)

			output = append(output, piece...)

			gModel.Forward(next, pos)
			pos++
			genCount++

			if pos >= gModel.Config.SeqLen {
				break
			}
		}
		ch <- genResult{output, genCount}
	}()
	r := <-ch

	// Copy to C buffer
	if len(r.output) > maxOut {
		r.output = r.output[:maxOut]
	}
	if len(r.output) > 0 {
		cOutput := (*[1 << 30]byte)(unsafe.Pointer(outputC))[:len(r.output)+1 : len(r.output)+1]
		copy(cOutput, r.output)
		cOutput[len(r.output)] = 0
	} else {
		cOutput := (*[1]byte)(unsafe.Pointer(outputC))
		cOutput[0] = 0
	}

	return C.int(r.genCount)
}

// ============================================================
// Tokenization
// ============================================================

//export wtf_encode
func wtf_encode(textC *C.char, idsOut *C.int, maxTokens C.int) C.int {
	if gTokenizer == nil {
		return 0
	}
	text := C.GoString(textC)
	ids := gTokenizer.Encode(text, false)

	max := int(maxTokens)
	if len(ids) > max {
		ids = ids[:max]
	}

	out := (*[1 << 20]C.int)(unsafe.Pointer(idsOut))[:len(ids):len(ids)]
	for i, id := range ids {
		out[i] = C.int(id)
	}
	return C.int(len(ids))
}

//export wtf_decode_token
func wtf_decode_token(id C.int, buf *C.char, bufLen C.int) C.int {
	if gTokenizer == nil || bufLen <= 0 {
		return 0
	}
	piece := gTokenizer.DecodeToken(int(id))
	maxLen := int(bufLen) - 1
	if len(piece) > maxLen {
		piece = piece[:maxLen]
	}
	if len(piece) > 0 {
		cBuf := (*[1 << 20]byte)(unsafe.Pointer(buf))[:len(piece)+1 : len(piece)+1]
		copy(cBuf, piece)
		cBuf[len(piece)] = 0
	} else {
		*(*byte)(unsafe.Pointer(buf)) = 0
	}
	return C.int(len(piece))
}

// ============================================================
// State queries
// ============================================================

//export wtf_get_vocab_size
func wtf_get_vocab_size() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.VocabSize)
}

//export wtf_get_dim
func wtf_get_dim() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.EmbedDim)
}

//export wtf_get_seq_len
func wtf_get_seq_len() C.int {
	if gModel == nil {
		return 0
	}
	return C.int(gModel.Config.SeqLen)
}

// ============================================================
// Required for c-shared build
// ============================================================

func main() {}
