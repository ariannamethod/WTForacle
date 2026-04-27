package wtf

// model.go — LLaMA-family forward pass for WTForacle (SmolLM2 360M).
//
// SmolLM2 360M architecture:
//   32 layers, 960 embed, 15 heads, 5 KV heads (GQA), 64 head_dim
//   2560 intermediate (gate_proj + up_proj + down_proj, SwiGLU)
//   RoPE theta=100000, RMSNorm eps=1e-5, no attention bias
//   Vocab 49152 (byte-level BPE)
//
// Weights are dequantized to float32 once at load time via the vendored
// notorch kernels (../ariannamethod/wtf_kernels.c). All matvec calls in
// the hot path go straight to cblas_sgemv via Apple Accelerate / OpenBLAS.

import (
	"fmt"
	"math"
	"runtime"
)

// LlamaModel is a loaded LLaMA-arch model ready for inference.
type LlamaModel struct {
	Config  LlamaConfig
	Weights LlamaWeights
	State   LlamaState
}

// LlamaConfig holds model dimensions.
type LlamaConfig struct {
	NumLayers  int
	EmbedDim   int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	VocabSize  int
	SeqLen     int
	IntermSize int
	RMSNormEps float32
	RopeTheta  float32
	// QKPermuted — convert_hf_to_gguf.py interleaves Q/K halves for
	// LLaMA-arch models. We un-permute after matmul so half-split RoPE works.
	QKPermuted bool
}

// LlamaWeights holds all weight tensors as contiguous float32 slices.
type LlamaWeights struct {
	TokenEmbed []float32 // [vocab, dim]
	OutputNorm []float32 // [dim]
	Output     []float32 // [vocab, dim] — may alias TokenEmbed when tied

	Layers []LlamaLayerWeights
}

type LlamaLayerWeights struct {
	AttnNorm []float32 // [dim]
	FFNNorm  []float32 // [dim]

	WQ []float32 // [n_heads*head_dim, dim]
	WK []float32 // [n_kv_heads*head_dim, dim]
	WV []float32 // [n_kv_heads*head_dim, dim]
	WO []float32 // [dim, n_heads*head_dim]

	BQ []float32 // optional — nil for SmolLM2
	BK []float32
	BV []float32
	BO []float32

	WGate []float32 // [interm, dim]
	WUp   []float32 // [interm, dim]
	WDown []float32 // [dim, interm]
}

// LlamaState holds runtime buffers + KV cache.
type LlamaState struct {
	X      []float32 // hidden state [dim]
	XB     []float32 // post-norm scratch [dim]
	XB2    []float32 // attention output scratch [dim]
	HB     []float32 // MLP gate scratch [interm]
	HB2    []float32 // MLP up scratch [interm]
	Q      []float32 // [n_heads*head_dim]
	K      []float32 // [n_kv_heads*head_dim]
	V      []float32 // [n_kv_heads*head_dim]
	Att    []float32 // [n_heads*seq_len]
	Logits []float32 // [vocab]

	KeyCache   []float32 // [layers*seq_len*kv_dim]
	ValueCache []float32

	CosCache []float32 // [seq_len*head_dim/2]
	SinCache []float32

	Pos int
}

// LoadLlamaModel builds a LlamaModel from a parsed GGUF file. Heavy: this
// dequantizes every weight tensor to float32 (cgo → notorch kernel) before
// returning.
func LoadLlamaModel(gguf *GGUFFile) (*LlamaModel, error) {
	m := gguf.Meta

	cfg := LlamaConfig{
		NumLayers:  m.NumLayers,
		EmbedDim:   m.EmbedDim,
		NumHeads:   m.NumHeads,
		NumKVHeads: m.NumKVHeads,
		HeadDim:    m.HeadDim,
		VocabSize:  m.VocabSize,
		SeqLen:     m.SeqLen,
		IntermSize: m.IntermSize,
		RMSNormEps: m.RMSNormEps,
		RopeTheta:  m.RopeTheta,
	}
	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	}

	arch := "llama"
	if v, ok := m.KV["general.architecture"]; ok {
		if s, ok := v.(string); ok {
			arch = s
		}
	}
	cfg.QKPermuted = (arch == "llama")

	// Cap context to keep KV cache reasonable on small machines.
	if cfg.SeqLen > 2048 {
		fmt.Printf("[tongue/model] capping seq_len from %d to 2048\n", cfg.SeqLen)
		cfg.SeqLen = 2048
	}

	w, err := loadWeights(gguf, &cfg)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Drop the raw GGUF byte buffer — every tensor is now F32 and the
	// quantized blob just sits there eating ~220 MB of RSS.
	gguf.TensorData = nil
	runtime.GC()

	state := allocState(&cfg)
	precomputeRoPE(&state, &cfg)

	hasBias := w.Layers[0].BQ != nil
	fmt.Printf("[tongue/model] loaded: %d layers, %d dim, %d heads, %d kv_heads, %d vocab, bias=%v, qk_permuted=%v\n",
		cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize, hasBias, cfg.QKPermuted)

	return &LlamaModel{Config: cfg, Weights: *w, State: state}, nil
}

// loadWeights resolves every tensor in the GGUF and dequantizes it to F32.
func loadWeights(gguf *GGUFFile, cfg *LlamaConfig) (*LlamaWeights, error) {
	w := &LlamaWeights{}

	embData, embInfo, err := gguf.GetTensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	embCount := cfg.VocabSize * cfg.EmbedDim
	w.TokenEmbed, err = dequantToF32(embData, embInfo.Type, embCount)
	if err != nil {
		return nil, fmt.Errorf("token_embd dequant: %w", err)
	}

	w.OutputNorm, err = getF32Tensor(gguf, "output_norm.weight", cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("output_norm.weight: %w", err)
	}

	// Output (LM head) — may be tied to token embedding.
	if outData, outInfo, err := gguf.GetTensor("output.weight"); err == nil {
		fmt.Printf("[tongue/model] output.weight: type=%d\n", outInfo.Type)
		w.Output, err = dequantToF32(outData, outInfo.Type, embCount)
		if err != nil {
			return nil, fmt.Errorf("output dequant: %w", err)
		}
	} else {
		fmt.Printf("[tongue/model] output.weight not found, using tied embeddings\n")
		w.Output = w.TokenEmbed
	}

	w.Layers = make([]LlamaLayerWeights, cfg.NumLayers)
	dim := cfg.EmbedDim
	qDim := cfg.NumHeads * cfg.HeadDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	interm := cfg.IntermSize

	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		l := &w.Layers[i]

		l.AttnNorm, err = getF32Tensor(gguf, prefix+"attn_norm.weight", dim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_norm: %w", i, err)
		}
		l.FFNNorm, err = getF32Tensor(gguf, prefix+"ffn_norm.weight", dim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}

		if l.WQ, err = dequantTensor(gguf, prefix+"attn_q.weight", qDim*dim); err != nil {
			return nil, fmt.Errorf("layer %d attn_q: %w", i, err)
		}
		if l.WK, err = dequantTensor(gguf, prefix+"attn_k.weight", kvDim*dim); err != nil {
			return nil, fmt.Errorf("layer %d attn_k: %w", i, err)
		}
		if l.WV, err = dequantTensor(gguf, prefix+"attn_v.weight", kvDim*dim); err != nil {
			return nil, fmt.Errorf("layer %d attn_v: %w", i, err)
		}
		if l.WO, err = dequantTensor(gguf, prefix+"attn_output.weight", dim*qDim); err != nil {
			return nil, fmt.Errorf("layer %d attn_output: %w", i, err)
		}

		l.BQ, _ = getF32TensorOptional(gguf, prefix+"attn_q.bias", qDim)
		l.BK, _ = getF32TensorOptional(gguf, prefix+"attn_k.bias", kvDim)
		l.BV, _ = getF32TensorOptional(gguf, prefix+"attn_v.bias", kvDim)
		l.BO, _ = getF32TensorOptional(gguf, prefix+"attn_output.bias", dim)

		if l.WGate, err = dequantTensor(gguf, prefix+"ffn_gate.weight", interm*dim); err != nil {
			return nil, fmt.Errorf("layer %d ffn_gate: %w", i, err)
		}
		if l.WUp, err = dequantTensor(gguf, prefix+"ffn_up.weight", interm*dim); err != nil {
			return nil, fmt.Errorf("layer %d ffn_up: %w", i, err)
		}
		if l.WDown, err = dequantTensor(gguf, prefix+"ffn_down.weight", dim*interm); err != nil {
			return nil, fmt.Errorf("layer %d ffn_down: %w", i, err)
		}
	}

	return w, nil
}

// dequantTensor pulls the named tensor from GGUF and routes through the
// notorch dequant kernel.
func dequantTensor(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, err
	}
	return dequantToF32(data, info.Type, expectedSize)
}

// getF32Tensor — F32 / F16 / Q* tensor → []float32 of the expected size.
func getF32Tensor(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	return dequantTensor(gguf, name, expectedSize)
}

// getF32TensorOptional returns nil (no error) when the tensor is missing.
func getF32TensorOptional(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	if _, _, err := gguf.GetTensor(name); err != nil {
		return nil, nil
	}
	return getF32Tensor(gguf, name, expectedSize)
}

// allocState allocates all runtime buffers.
func allocState(cfg *LlamaConfig) LlamaState {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return LlamaState{
		X:          make([]float32, cfg.EmbedDim),
		XB:         make([]float32, cfg.EmbedDim),
		XB2:        make([]float32, cfg.EmbedDim),
		HB:         make([]float32, cfg.IntermSize),
		HB2:        make([]float32, cfg.IntermSize),
		Q:          make([]float32, cfg.NumHeads*cfg.HeadDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Att:        make([]float32, cfg.NumHeads*cfg.SeqLen),
		Logits:     make([]float32, cfg.VocabSize),
		KeyCache:   make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		ValueCache: make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		CosCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		SinCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
	}
}

func precomputeRoPE(s *LlamaState, cfg *LlamaConfig) {
	half := cfg.HeadDim / 2
	theta := float64(cfg.RopeTheta)
	for pos := 0; pos < cfg.SeqLen; pos++ {
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(cfg.HeadDim))
			angle := float64(pos) * freq
			s.CosCache[pos*half+i] = float32(math.Cos(angle))
			s.SinCache[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// applyRoPE rotates one head with the cached cos/sin.
// Half-split layout: vec[i] pairs with vec[i+half].
func applyRoPE(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	off := pos * half
	for i := 0; i < half; i++ {
		x0, x1 := vec[i], vec[i+half]
		c, si := s.CosCache[off+i], s.SinCache[off+i]
		vec[i] = x0*c - x1*si
		vec[i+half] = x0*si + x1*c
	}
}

// unpermuteQK reverses the convert_hf_to_gguf.py Q/K interleave so the
// half-split RoPE layout above is correct.
func unpermuteQK(vec []float32, nHeads, headDim int) {
	half := headDim / 2
	tmp := make([]float32, headDim)
	for h := 0; h < nHeads; h++ {
		base := h * headDim
		for i := 0; i < half; i++ {
			tmp[i] = vec[base+2*i]
			tmp[half+i] = vec[base+2*i+1]
		}
		copy(vec[base:base+headDim], tmp)
	}
}

func addBias(out, bias []float32) {
	if bias == nil {
		return
	}
	for i := range bias {
		out[i] += bias[i]
	}
}

// Forward runs one token through the transformer at position `pos`.
func (m *LlamaModel) Forward(token int, pos int) {
	cfg := &m.Config
	w := &m.Weights
	s := &m.State
	dim := cfg.EmbedDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	hd := cfg.HeadDim
	headGroup := cfg.NumHeads / cfg.NumKVHeads

	// Token embedding lookup — direct copy from the F32 table.
	copy(s.X, w.TokenEmbed[token*dim:token*dim+dim])

	attnScale := float32(1.0 / math.Sqrt(float64(hd)))

	for layer := 0; layer < cfg.NumLayers; layer++ {
		l := &w.Layers[layer]

		// Attention pre-norm
		RMSNormInto(s.XB, s.X, l.AttnNorm, cfg.RMSNormEps)

		// Q, K, V projections via BLAS sgemv
		sgemv(s.Q, l.WQ, s.XB, cfg.NumHeads*hd, dim)
		sgemv(s.K, l.WK, s.XB, cfg.NumKVHeads*hd, dim)
		sgemv(s.V, l.WV, s.XB, cfg.NumKVHeads*hd, dim)

		addBias(s.Q, l.BQ)
		addBias(s.K, l.BK)
		addBias(s.V, l.BV)

		if cfg.QKPermuted {
			unpermuteQK(s.Q, cfg.NumHeads, hd)
			unpermuteQK(s.K, cfg.NumKVHeads, hd)
		}

		// RoPE on Q and K
		for h := 0; h < cfg.NumHeads; h++ {
			applyRoPE(s.Q[h*hd:(h+1)*hd], pos, s, hd)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			applyRoPE(s.K[h*hd:(h+1)*hd], pos, s, hd)
		}

		// Store K, V into the cache for this position
		cacheOff := layer*cfg.SeqLen*kvDim + pos*kvDim
		copy(s.KeyCache[cacheOff:cacheOff+kvDim], s.K[:kvDim])
		copy(s.ValueCache[cacheOff:cacheOff+kvDim], s.V[:kvDim])

		// Multi-head attention with GQA. The KV cache for this layer is laid
		// out as [seq_len, kv_dim], and each head reads a [pos+1, head_dim]
		// strided sub-view. We feed those views straight to BLAS sgemv.
		layerBase := layer * cfg.SeqLen * kvDim
		for h := 0; h < cfg.NumHeads; h++ {
			kvh := h / headGroup
			qh := s.Q[h*hd : (h+1)*hd]
			att := s.Att[h*cfg.SeqLen : h*cfg.SeqLen+pos+1]

			// QK^T: att[pos+1] = K[pos+1, hd] @ qh[hd]
			kBase := layerBase + kvh*hd
			sgemvStrided(att, s.KeyCache[kBase:], kvDim, qh, pos+1, hd, false)
			for t := 0; t <= pos; t++ {
				att[t] *= attnScale
			}

			Softmax(att, pos+1)

			// att·V: xb[hd] = V[pos+1, hd]^T @ att[pos+1]
			xb := s.XB2[h*hd : (h+1)*hd]
			vBase := layerBase + kvh*hd
			sgemvStrided(xb, s.ValueCache[vBase:], kvDim, att, pos+1, hd, true)
		}

		// Output projection + residual
		sgemv(s.XB, l.WO, s.XB2, dim, dim)
		addBias(s.XB, l.BO)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}

		// MLP pre-norm
		RMSNormInto(s.XB, s.X, l.FFNNorm, cfg.RMSNormEps)

		// SwiGLU: silu(gate(x)) * up(x), then down(...)
		sgemv(s.HB, l.WGate, s.XB, cfg.IntermSize, dim)
		sgemv(s.HB2, l.WUp, s.XB, cfg.IntermSize, dim)
		for i := 0; i < cfg.IntermSize; i++ {
			s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
		}
		sgemv(s.XB, l.WDown, s.HB, dim, cfg.IntermSize)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	// Final norm + LM head
	RMSNorm(s.X, w.OutputNorm, cfg.RMSNormEps)
	sgemv(s.Logits, w.Output, s.X, cfg.VocabSize, dim)
}

// Reset clears the KV cache and position for a fresh generation.
func (m *LlamaModel) Reset() {
	for i := range m.State.KeyCache {
		m.State.KeyCache[i] = 0
	}
	for i := range m.State.ValueCache {
		m.State.ValueCache[i] = 0
	}
	m.State.Pos = 0
}
