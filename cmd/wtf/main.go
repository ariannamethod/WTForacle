package main

// cmd/wtf/main.go — WTForacle native Go CLI + REPL.
//
//   wtf -prompt "explain ai"            # one-shot, prints to stdout
//   echo "stuff" | wtf                  # stdin one-shot
//   wtf                                  # interactive REPL with limpha memory
//
// No Python. No ctypes. Just Go, cgo into notorch, and a SQLite file under
// ~/.wtforacle/ for limpha memory.

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"wtforacle/wtf"
)

const banner = "" +
	"============================================================\n" +
	"  WTFORACLE\n" +
	"  the reddit oracle nobody asked for\n" +
	"  WTForacle v3 (SmolLM2 360M, Q4_0 → notorch sgemv)\n" +
	"============================================================\n"

const systemPrompt = "" +
	"you are wtforacle, a cynical reddit commenter. " +
	"you love to argue for fun, and when you argue you write more. " +
	"if you say something helpful, always end with sarcasm. " +
	"a weird answer is always better than a boring one. " +
	"talk like reddit: bro, tbh, ngl, imo. " +
	"if you apologize, do it with sarcasm. " +
	"if someone asks for code, give the code but call them lazy."

func main() {
	weightsFlag := flag.String("weights", "", "path to GGUF weights (default: ./wtfweights/wtf360_v2_q4_0.gguf)")
	prompt := flag.String("prompt", "", "one-shot prompt (omit to enter REPL)")
	maxTokens := flag.Int("max", 200, "max tokens to generate")
	temp := flag.Float64("temp", 0.9, "sampling temperature")
	topP := flag.Float64("top-p", 0.9, "top-p (nucleus) threshold")
	rawFlag := flag.Bool("raw", false, "skip system prompt (raw mode)")
	trollFlag := flag.Bool("troll", false, "trolling mode (3 candidates, spiciest wins)")
	flag.Parse()

	weights := *weightsFlag
	if weights == "" {
		exe, _ := os.Executable()
		weights = filepath.Join(filepath.Dir(exe), "wtfweights", "wtf360_v2_q4_0.gguf")
		if _, err := os.Stat(weights); err != nil {
			weights = "wtfweights/wtf360_v2_q4_0.gguf"
		}
	}

	model, tokenizer := loadModel(weights)

	// One-shot mode: explicit -prompt only. Stdin is REPL by default so that
	// piped multi-line scripts like `printf '/stats\n/quit\n' | wtforacle`
	// behave the same as typing into a TTY.
	if *prompt != "" {
		out := generateOnce(model, tokenizer, *prompt, *maxTokens, float32(*temp), float32(*topP),
			!*rawFlag, *trollFlag)
		fmt.Println(out)
		return
	}

	repl(model, tokenizer, *maxTokens, *temp, *topP)
}

func loadModel(path string) (*wtf.LlamaModel, *wtf.Tokenizer) {
	fmt.Fprintf(os.Stderr, "[wtf] loading %s\n", path)
	gguf, err := wtf.LoadGGUF(path)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading GGUF: %v\n", err)
		os.Exit(1)
	}
	model, err := wtf.LoadLlamaModel(gguf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error loading model: %v\n", err)
		os.Exit(1)
	}
	tok := wtf.NewTokenizer(&gguf.Meta)
	fmt.Fprintf(os.Stderr, "[wtf] ready: %d layers, %d dim, %d vocab\n",
		model.Config.NumLayers, model.Config.EmbedDim, model.Config.VocabSize)
	return model, tok
}

// ─────────────────────────────────────────────────────────────────────────────
// Generation — single call

func buildPrompt(text string, useSystem bool) string {
	if useSystem {
		return systemPrompt + "\n### Question: " + text + "\n### Answer:"
	}
	return "### Question: " + text + "\n### Answer:"
}

func generateOnce(model *wtf.LlamaModel, tok *wtf.Tokenizer, userPrompt string,
	maxTokens int, temp, topP float32, useSystem, troll bool) string {

	if troll {
		text, _, _ := generateTroll(model, tok, userPrompt, maxTokens, useSystem)
		return text
	}
	full := buildPrompt(userPrompt, useSystem)
	return generate(model, tok, full, maxTokens, temp, topP)
}

// generate runs one decode pass starting from `prompt`, returning the
// generated text. Reuses sampling buffers across tokens.
func generate(model *wtf.LlamaModel, tok *wtf.Tokenizer, prompt string,
	maxTokens int, temp, topP float32) string {

	model.Reset()
	repPenalty := float32(1.15)
	repWindow := 64

	var allTokens []int
	if tok.BosID >= 0 && tok.BosID != tok.EosID {
		allTokens = append(allTokens, tok.BosID)
	}
	allTokens = append(allTokens, tok.Encode(prompt, false)...)

	pos := 0
	for _, t := range allTokens {
		model.Forward(t, pos)
		pos++
		if pos >= model.Config.SeqLen-1 {
			break
		}
	}

	sb := wtf.NewSampleBuffers(model.Config.VocabSize)
	vocab := model.Config.VocabSize

	var out []byte
	graceLimit := 32
	inGrace := false
	recent := make([]int, 0, repWindow)
	counts := make(map[int]int, 64)

	for i := 0; i < maxTokens+graceLimit; i++ {
		if i >= maxTokens && !inGrace {
			inGrace = true
		}
		if inGrace && len(out) > 0 {
			last := out[len(out)-1]
			if last == '.' || last == '!' || last == '?' || last == '\n' {
				break
			}
		}

		// Repetition penalty (presence-based, sliding window)
		for _, t := range recent {
			lg := model.State.Logits[t]
			if lg > 0 {
				model.State.Logits[t] = lg / repPenalty
			} else {
				model.State.Logits[t] = lg * repPenalty
			}
		}

		var next int
		if topP < 1.0 {
			next = wtf.SampleTopP(model.State.Logits, vocab, temp, topP, sb)
		} else {
			next = wtf.SampleTopK(model.State.Logits, vocab, temp, 50, sb)
		}

		counts[next]++
		recent = append(recent, next)
		if len(recent) > repWindow {
			leaving := recent[0]
			counts[leaving]--
			if counts[leaving] <= 0 {
				delete(counts, leaving)
			}
			recent = recent[1:]
		}

		if next == tok.EosID {
			break
		}

		// Cycle detection: last 8 tokens match the 8 before that
		if len(recent) >= 16 {
			n := len(recent)
			cycle := true
			for k := 0; k < 8; k++ {
				if recent[n-1-k] != recent[n-9-k] {
					cycle = false
					break
				}
			}
			if cycle {
				break
			}
		}

		out = append(out, tok.DecodeToken(next)...)
		model.Forward(next, pos)
		pos++
		if pos >= model.Config.SeqLen {
			break
		}
	}

	return string(out)
}

// ─────────────────────────────────────────────────────────────────────────────
// Trolling mode

// generateTroll runs three decodes at temps 0.9 / 1.0 / 1.1 and returns the
// spiciest one. Decodes serialize because the model has shared state.
func generateTroll(model *wtf.LlamaModel, tok *wtf.Tokenizer,
	userPrompt string, maxTokens int, useSystem bool) (string, float32, string) {

	full := buildPrompt(userPrompt, useSystem)
	temps := []float32{0.9, 1.0, 1.1}
	type cand struct {
		text  string
		temp  float32
		score float64
	}
	cands := make([]cand, 0, len(temps))
	for _, t := range temps {
		text := generate(model, tok, full, maxTokens, t, 1.0)
		cands = append(cands, cand{text: text, temp: t, score: scoreTroll(text)})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].score > cands[j].score })

	parts := make([]string, 0, len(cands))
	bestTemp := cands[0].temp
	for _, c := range cands {
		mark := ""
		if c.temp == bestTemp {
			mark = "*"
		}
		parts = append(parts, fmt.Sprintf("t=%.1f:%.0f%s", c.temp, c.score, mark))
	}
	report := strings.Join(parts, " | ")
	return cands[0].text, cands[0].temp, report
}

// scoreTroll mirrors wtforacle.py:_score_response — same constants so saved
// /troll choices stay comparable across versions.
func scoreTroll(text string) float64 {
	if len(strings.TrimSpace(text)) < 5 {
		return -100
	}
	score := 0.0
	words := strings.Fields(text)
	if w := len(words); w < 80 {
		score += float64(w) * 0.5
	} else {
		score += 80 * 0.5
	}
	lower := strings.ToLower(text)
	for _, s := range []string{"bro", "tbh", "ngl", "imo", "lmao", "lol", "bruh",
		"nah", "fr", "literally", "actually", "honestly",
		"ok so", "look", "the thing is", "imagine"} {
		score += float64(strings.Count(lower, s)) * 3
	}
	score += float64(strings.Count(text, "?")) * 2
	score += float64(strings.Count(text, "!")) * 1.5
	score += float64(strings.Count(text, "...")) * 2

	alpha := 0
	lowerAlpha := 0
	for _, r := range text {
		if unicode.IsLetter(r) {
			alpha++
			if unicode.IsLower(r) {
				lowerAlpha++
			}
		}
	}
	if alpha > 0 && float64(lowerAlpha)/float64(alpha) > 0.9 {
		score += 5
	}
	for _, b := range []string{"as an ai", "i cannot", "i apologize",
		"how can i help", "i'd be happy to", "great question"} {
		if strings.Contains(lower, b) {
			score -= 20
		}
	}
	return score
}

// ─────────────────────────────────────────────────────────────────────────────
// Interactive REPL

func repl(model *wtf.LlamaModel, tok *wtf.Tokenizer, defaultMax int, defaultTemp, defaultTopP float64) {
	fmt.Println(banner)

	mem, err := wtf.OpenLimpha()
	if err != nil {
		fmt.Printf("  memory: offline (%v)\n", err)
		mem = nil
	} else {
		fmt.Println("  memory: online (limpha)")
		defer mem.Close()
	}

	fmt.Println("Commands: /quit, /tokens N, /temp T, /raw, /troll")
	if mem != nil {
		fmt.Println("Memory:   /recall QUERY, /recent, /stats")
	}
	fmt.Println()

	maxTokens := defaultMax
	temp := float32(defaultTemp)
	topP := float32(defaultTopP)
	useSystem := true
	troll := false

	r := bufio.NewReader(os.Stdin)
	for {
		fmt.Print("You: ")
		line, err := r.ReadString('\n')
		if err == io.EOF {
			fmt.Println("\nlater loser")
			return
		} else if err != nil {
			fmt.Printf("\nerror: %v\n", err)
			return
		}
		input := strings.TrimSpace(line)
		if input == "" {
			continue
		}

		lower := strings.ToLower(input)
		switch {
		case lower == "/quit", lower == "/exit", lower == "/q":
			if mem != nil {
				if s, err := mem.Stats(); err == nil && s.TotalConversations > 0 {
					fmt.Printf("(%d conversations remembered)\n", s.TotalConversations)
				}
			}
			fmt.Println("later loser")
			return

		case strings.HasPrefix(lower, "/tokens "):
			if n, err := strconv.Atoi(strings.TrimSpace(input[8:])); err == nil {
				maxTokens = n
				fmt.Printf("Max tokens set to %d\n", maxTokens)
			} else {
				fmt.Println("Usage: /tokens N")
			}
			continue

		case strings.HasPrefix(lower, "/temp "):
			if t, err := strconv.ParseFloat(strings.TrimSpace(input[6:]), 32); err == nil {
				temp = float32(t)
				fmt.Printf("Temperature set to %.2f\n", temp)
			} else {
				fmt.Println("Usage: /temp T")
			}
			continue

		case lower == "/raw":
			useSystem = !useSystem
			if useSystem {
				fmt.Println("System prompt: ON")
			} else {
				fmt.Println("System prompt: OFF (raw mode)")
			}
			continue

		case lower == "/troll":
			troll = !troll
			if troll {
				fmt.Println("Trolling mode: ON (3 candidates, best wins)")
			} else {
				fmt.Println("Trolling mode: OFF")
			}
			continue

		case strings.HasPrefix(lower, "/recall ") && mem != nil:
			query := strings.TrimSpace(input[8:])
			if query == "" {
				fmt.Println("Usage: /recall QUERY")
				continue
			}
			results, _ := mem.Search(query, 5)
			if len(results) == 0 {
				fmt.Println("nothing found. memory is empty or your query sucks.")
			} else {
				for _, r := range results {
					fmt.Printf("  [%d] You: %s\n", r.ID, r.Prompt)
					fmt.Printf("       WTF: %s...\n\n", trunc(r.Response, 120))
				}
			}
			continue

		case lower == "/recent" && mem != nil:
			convs, _ := mem.Recent(5)
			if len(convs) == 0 {
				fmt.Println("no memory yet. start talking.")
			} else {
				for _, c := range convs {
					fmt.Printf("  [%d] You: %s\n", c.ID, c.Prompt)
					fmt.Printf("       WTF: %s...\n\n", trunc(c.Response, 120))
				}
			}
			continue

		case lower == "/stats" && mem != nil:
			s, err := mem.Stats()
			if err != nil {
				fmt.Printf("stats error: %v\n", err)
				continue
			}
			fmt.Printf("  conversations: %d\n", s.TotalConversations)
			fmt.Printf("  sessions: %d\n", s.TotalSessions)
			fmt.Printf("  avg quality: %.3f\n", s.AvgQuality)
			fmt.Printf("  db: %s\n", s.DBPath)
			fmt.Printf("  size: %.1f KB\n", float64(s.DBSizeBytes)/1024.0)
			continue
		}

		// Generation
		fmt.Print("\nWTForacle: ")
		var response string
		if troll {
			text, _, report := generateTroll(model, tok, input, maxTokens, useSystem)
			response = text
			fmt.Println(strings.TrimSpace(text))
			fmt.Printf("  [%s]\n", report)
		} else {
			full := buildPrompt(input, useSystem)
			response = generate(model, tok, full, maxTokens, temp, topP)
			fmt.Println(strings.TrimSpace(response))
		}
		fmt.Println()

		if mem != nil && strings.TrimSpace(response) != "" {
			_, _ = mem.Store(input, response, float64(temp))
		}
	}
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}
