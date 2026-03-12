package main

// cmd/wtf/main.go — native Go CLI for WTForacle inference.
//
// Usage:
//   wtf -weights model.gguf -prompt "why does AI suck"
//   echo "explain kubernetes" | wtf -weights model.gguf
//
// No frameworks. No Python. Just vibes and logits.

import (
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"wtforacle/wtf"
)

const defaultSystem = "You are WTForacle, a mass of cynical oracular energy. You give short, sharp answers dripping with sarcasm. You hate corporate buzzwords, mass hype, and anyone who thinks AI will save humanity. Keep it real, keep it brief."

func main() {
	weights := flag.String("weights", "", "path to GGUF weights file (required)")
	prompt := flag.String("prompt", "", "input prompt (default: read from stdin)")
	maxTokens := flag.Int("max", 200, "max tokens to generate")
	temp := flag.Float64("temp", 0.9, "sampling temperature")
	topP := flag.Float64("top-p", 0.9, "top-p (nucleus) sampling threshold")
	system := flag.String("system", defaultSystem, "system prompt")
	troll := flag.Bool("troll", false, "trolling mode: generate 3 candidates, pick spiciest")
	flag.Parse()

	if *weights == "" {
		fmt.Fprintln(os.Stderr, "error: -weights is required")
		flag.Usage()
		os.Exit(1)
	}

	// Get prompt from flag or stdin
	input := *prompt
	if input == "" {
		data, err := io.ReadAll(os.Stdin)
		if err != nil {
			fmt.Fprintf(os.Stderr, "error reading stdin: %v\n", err)
			os.Exit(1)
		}
		input = strings.TrimSpace(string(data))
	}
	if input == "" {
		fmt.Fprintln(os.Stderr, "error: no prompt provided (use -prompt or pipe to stdin)")
		os.Exit(1)
	}

	// Load model
	fmt.Fprintf(os.Stderr, "[wtf] loading %s\n", *weights)
	gguf, err := wtf.LoadGGUF(*weights)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading GGUF: %v\n", err)
		os.Exit(1)
	}
	model, err := wtf.LoadLlamaModel(gguf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading model: %v\n", err)
		os.Exit(1)
	}
	tokenizer := wtf.NewTokenizer(&gguf.Meta)
	fmt.Fprintf(os.Stderr, "[wtf] ready: %d layers, %d dim, %d vocab\n",
		model.Config.NumLayers, model.Config.EmbedDim, model.Config.VocabSize)

	// Build full prompt with system context
	fullPrompt := *system + "\n\nUser: " + input + "\nAssistant:"

	if *troll {
		// Trolling mode: generate 3 candidates, pick longest (spiciest)
		best := ""
		for i := 0; i < 3; i++ {
			out := generate(model, tokenizer, fullPrompt, *maxTokens, float32(*temp), float32(*topP))
			if len(out) > len(best) {
				best = out
			}
		}
		fmt.Print(best)
	} else {
		out := generate(model, tokenizer, fullPrompt, *maxTokens, float32(*temp), float32(*topP))
		fmt.Print(out)
	}
}

// generate runs inference and returns the generated text.
func generate(model *wtf.LlamaModel, tokenizer *wtf.Tokenizer, prompt string, maxTokens int, temp, topP float32) string {
	model.Reset()

	// Repetition penalty settings
	repPenalty := float32(1.15)
	repWindow := 64

	// Build token sequence: [optional BOS] + prompt tokens
	var allTokens []int
	if tokenizer.BosID >= 0 && tokenizer.BosID != tokenizer.EosID {
		allTokens = append(allTokens, tokenizer.BosID)
	}
	promptTokens := tokenizer.Encode(prompt, false)
	allTokens = append(allTokens, promptTokens...)

	// Prefill: feed prompt tokens through transformer
	pos := 0
	for _, tok := range allTokens {
		model.Forward(tok, pos)
		pos++
		if pos >= model.Config.SeqLen-1 {
			break
		}
	}

	// Allocate sampling buffers
	sb := wtf.NewSampleBuffers(model.Config.VocabSize)

	// Generate
	var output []byte
	graceLimit := 32
	inGrace := false
	recentTokens := make([]int, 0, repWindow)
	tokenCounts := make(map[int]int, 64)
	vocab := model.Config.VocabSize

	for i := 0; i < maxTokens+graceLimit; i++ {
		if i >= maxTokens && !inGrace {
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
				logit := model.State.Logits[tok]
				if logit > 0 {
					model.State.Logits[tok] = logit / repPenalty
				} else {
					model.State.Logits[tok] = logit * repPenalty
				}
			}
		}

		// Sample next token
		var next int
		if topP < 1.0 {
			next = wtf.SampleTopP(model.State.Logits, vocab, temp, topP, sb)
		} else {
			next = wtf.SampleTopK(model.State.Logits, vocab, temp, 50, sb)
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
		if next == tokenizer.EosID {
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
				break
			}
		}

		piece := tokenizer.DecodeToken(next)
		output = append(output, piece...)

		model.Forward(next, pos)
		pos++

		if pos >= model.Config.SeqLen {
			break
		}
	}

	return string(output)
}
