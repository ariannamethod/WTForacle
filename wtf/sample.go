package wtf

// sample.go — zero-alloc sampling for WTForacle inference.
//
// Top-K and Top-P (nucleus) sampling with pre-allocated buffers.
// No allocations in the hot path — all buffers reused per token.

import (
	"math"
	"math/rand"
	"sort"
	"time"
)

// SampleBuffers holds pre-allocated buffers for sampling (no alloc per token)
type SampleBuffers struct {
	// For topP: candidates sorted by probability
	candidates []idxProb // [vocab]

	// For topK
	topIdx   []int32
	topVal   []float32
	topProbs []float32

	// RNG for sampling
	RNG *rand.Rand
}

type idxProb struct {
	idx  int
	prob float32
}

// NewSampleBuffers creates pre-allocated sampling buffers for the given vocab size.
func NewSampleBuffers(vocab int) *SampleBuffers {
	return &SampleBuffers{
		candidates: make([]idxProb, vocab),
		topIdx:     make([]int32, 50), // topK=50
		topVal:     make([]float32, 50),
		topProbs:   make([]float32, 50),
		RNG:        rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// SampleTopK samples from top-K logits with temperature scaling.
func SampleTopK(logits []float32, vocab int, temp float32, topK int, sb *SampleBuffers) int {
	if temp <= 0 {
		return Argmax(logits, vocab)
	}
	if topK > vocab {
		topK = vocab
	}

	// Reset top-k buffers
	for i := 0; i < topK; i++ {
		sb.topIdx[i] = -1
		sb.topVal[i] = -1e30
	}

	// Find top-k indices (single pass, no allocation)
	for i := 0; i < vocab; i++ {
		if logits[i] > sb.topVal[topK-1] {
			sb.topIdx[topK-1] = int32(i)
			sb.topVal[topK-1] = logits[i]
			for j := topK - 1; j > 0 && sb.topVal[j] > sb.topVal[j-1]; j-- {
				sb.topIdx[j], sb.topIdx[j-1] = sb.topIdx[j-1], sb.topIdx[j]
				sb.topVal[j], sb.topVal[j-1] = sb.topVal[j-1], sb.topVal[j]
			}
		}
	}

	// Softmax over top-k (reuse probs buffer)
	maxVal := sb.topVal[0]
	var sum float32
	for i := 0; i < topK; i++ {
		if sb.topIdx[i] < 0 {
			break
		}
		sb.topProbs[i] = float32(math.Exp(float64((sb.topVal[i] - maxVal) / temp)))
		sum += sb.topProbs[i]
	}

	// Sample
	r := sb.RNG.Float32() * sum
	var cdf float32
	for i := 0; i < topK; i++ {
		cdf += sb.topProbs[i]
		if r <= cdf {
			return int(sb.topIdx[i])
		}
	}
	return int(sb.topIdx[0])
}

// SampleTopP samples using nucleus (top-p) sampling with temperature.
func SampleTopP(logits []float32, vocab int, temp float32, topP float32, sb *SampleBuffers) int {
	if temp <= 0 {
		return Argmax(logits, vocab)
	}

	// Apply temperature and compute softmax (reuse sb.candidates — zero alloc)
	maxVal := logits[0]
	for i := 1; i < vocab; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	var sum float32
	for i := 0; i < vocab; i++ {
		p := float32(math.Exp(float64((logits[i] - maxVal) / temp)))
		sb.candidates[i].idx = i
		sb.candidates[i].prob = p
		sum += p
	}

	// Normalize
	invSum := float32(1.0) / sum
	for i := 0; i < vocab; i++ {
		sb.candidates[i].prob *= invSum
	}

	// Sort by probability descending (cache-friendly struct slice)
	sort.Slice(sb.candidates[:vocab], func(i, j int) bool {
		return sb.candidates[i].prob > sb.candidates[j].prob
	})

	// Find nucleus and sample
	var cumsum float32
	for i := 0; i < vocab; i++ {
		cumsum += sb.candidates[i].prob
		if cumsum >= topP {
			r := sb.RNG.Float32() * cumsum
			var cdf float32
			for j := 0; j <= i; j++ {
				cdf += sb.candidates[j].prob
				if r <= cdf {
					return sb.candidates[j].idx
				}
			}
			return sb.candidates[0].idx
		}
	}
	return sb.candidates[0].idx
}

// Argmax returns the index of the largest value in logits[:n].
func Argmax(logits []float32, n int) int {
	best := 0
	for i := 1; i < n; i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}
