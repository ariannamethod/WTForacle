package wtf

// model.go — LLaMA-family forward pass for WTForacle (SmolLM2 360M).
//
// SmolLM2 360M architecture:
//   32 layers, 960 embed, 15 heads, 5 KV heads (GQA), 64 head_dim
//   2560 intermediate (gate_proj + up_proj + down_proj, SwiGLU)
//   RoPE theta=100000, RMSNorm eps=1e-5, no attention bias
//   Vocab 49152 (byte-level BPE)
//
// Layer weight matrices are kept in their packed GGUF encoding and matvec'd
// straight from packed bytes via notorch's nt_qmatvec (no dense-f32 blow-up) —
// see QW / loadQW below. Token embeddings and norms stay f32 (the embedding
// lookup needs f32 rows). On SmolLM2-360M Q4_0 this cut peak RSS 1600 MB -> 588 MB
// (neo, A18 Pro) with byte-identical greedy output vs the old f32 path.

import (
	"fmt"
	"math"
	"runtime"
	"strings"
)

// LlamaModel is a loaded LLaMA-arch model ready for inference.
type LlamaModel struct {
	Config  LlamaConfig
	Weights LlamaWeights
	State   LlamaState
}

// LlamaConfig holds model dimensions.
type LlamaConfig struct {
	Architecture   string
	BaseModelLabel string
	NumLayers      int
	EmbedDim       int
	NumHeads       int
	NumKVHeads     int
	HeadDim        int
	VocabSize      int
	SeqLen         int
	IntermSize     int
	RMSNormEps     float32
	RopeTheta      float32
	// QKPermuted — convert_hf_to_gguf.py interleaves Q/K halves for
	// LLaMA-arch models. We un-permute after matmul so half-split RoPE works.
	QKPermuted bool
}

// LlamaWeights holds transformer weights. The large vocab tables are kept
// packed when possible: embedding lookup dequantizes one row, and the tied
// LM head matvecs directly from packed bytes.
type LlamaWeights struct {
	TokenEmbed QTable    // [vocab, dim]
	OutputNorm []float32 // [dim]
	Output     QTable    // [vocab, dim] — may alias TokenEmbed when tied

	Layers []LlamaLayerWeights
}

type LlamaLayerWeights struct {
	AttnNorm []float32 // [dim]
	FFNNorm  []float32 // [dim]

	WQ QW // [n_heads*head_dim, dim]
	WK QW // [n_kv_heads*head_dim, dim]
	WV QW // [n_kv_heads*head_dim, dim]
	WO QW // [dim, n_heads*head_dim]

	BQ []float32 // optional — nil for SmolLM2
	BK []float32
	BV []float32
	BO []float32

	QNorm []float32 // optional Qwen3 per-head RMSNorm [head_dim]
	KNorm []float32

	WGate QW // [interm, dim]
	WUp   QW // [interm, dim]
	WDown QW // [dim, interm]
}

// QW is a weight matrix [M,K] kept in its packed GGUF encoding (Packed != nil) so
// it is never blown up to dense f32 in RAM. When the dtype has no packed kernel,
// F32 holds the dequantized fallback instead.
type QW struct {
	Packed []byte    // packed GGUF bytes (owned copy), nil if dequantized
	F32    []float32 // dequantized fallback, nil if packed
	Dtype  int
	M, K   int
}

// matvec computes out[M] = W[M,K] @ x[K] — packed via notorch nt_qmatvec when the
// weight is packed, else cblas sgemv on the f32 fallback.
func (w *QW) matvec(out, x []float32) {
	if w.Packed != nil {
		if useQmatvecI8 && qmatvecI8(out, w.Packed, w.Dtype, x, w.M, w.K) {
			return
		}
		qmatvec(out, w.Packed, w.Dtype, x, w.M, w.K)
		return
	}
	sgemv(out, w.F32, x, w.M, w.K)
}

func qmatvecSupported(dt int) bool {
	switch dt {
	case dtypeF32, dtypeF16, dtypeQ4_0, dtypeQ5_0, dtypeQ8_0, dtypeQ4_K, dtypeQ6_K:
		return true
	}
	return false
}

func packedMatvecSupported(dt int, k int) bool {
	if !qmatvecSupported(dt) {
		return false
	}
	blockElems := ggmlBlockElements(uint32(dt))
	return blockElems > 0 && k%blockElems == 0
}

func qtablePackedSupported(dt int, cols int) bool {
	return dt != dtypeF32 && packedMatvecSupported(dt, cols)
}

// QTable is a row-major [Rows, Cols] table, usually token_embd/output. It stays
// packed for quantized/F16 dtypes so WTForacle does not materialize the whole
// vocab x dim table as f32.
type QTable struct {
	Packed   []byte
	F32      []float32
	Dtype    int
	Rows     int
	Cols     int
	RowBytes int
}

func (t *QTable) lookup(out []float32, row int) {
	if row < 0 || row >= t.Rows {
		panic(fmt.Sprintf("QTable.lookup: row=%d rows=%d", row, t.Rows))
	}
	if len(out) < t.Cols {
		panic(fmt.Sprintf("QTable.lookup: out len=%d cols=%d", len(out), t.Cols))
	}
	if t.Packed != nil {
		start := row * t.RowBytes
		end := start + t.RowBytes
		if err := dequantToF32Into(t.Packed[start:end], uint32(t.Dtype), out, t.Cols); err != nil {
			panic(fmt.Sprintf("QTable.lookup: %v", err))
		}
		return
	}
	copy(out, t.F32[row*t.Cols:(row+1)*t.Cols])
}

func (t *QTable) matvec(out, x []float32) {
	if t.Packed != nil {
		if useQmatvecI8 && qmatvecI8(out, t.Packed, t.Dtype, x, t.Rows, t.Cols) {
			return
		}
		if qmatvec(out, t.Packed, t.Dtype, x, t.Rows, t.Cols) {
			return
		}
		panic(fmt.Sprintf("QTable.matvec: unsupported packed dtype=%d cols=%d", t.Dtype, t.Cols))
	}
	sgemv(out, t.F32, x, t.Rows, t.Cols)
}

func qtableRowBytes(dtype uint32, cols int) (int, error) {
	blockSize := ggmlBlockSize(dtype)
	blockElems := ggmlBlockElements(dtype)
	if blockSize <= 0 || blockElems <= 0 {
		return 0, fmt.Errorf("unsupported dtype %d", dtype)
	}
	if cols%blockElems != 0 {
		return 0, fmt.Errorf("cols=%d is not divisible by block elements=%d for dtype %d", cols, blockElems, dtype)
	}
	return (cols / blockElems) * blockSize, nil
}

// loadQW loads the [m,k] matrix named `name`, kept PACKED when nt_qmatvec supports
// its dtype (bytes copied so the GGUF blob can be freed), else dequantized to f32.
func loadQW(gguf *GGUFFile, name string, m, k int) (QW, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return QW{}, err
	}
	if err := validateMatrixTensor(info, name, m, k); err != nil {
		return QW{}, err
	}
	dt := int(info.Type)
	if packedMatvecSupported(dt, k) {
		packed := make([]byte, len(data))
		copy(packed, data)
		return QW{Packed: packed, Dtype: dt, M: m, K: k}, nil
	}
	f32, err := dequantToF32(data, info.Type, m*k)
	if err != nil {
		return QW{}, err
	}
	return QW{F32: f32, Dtype: dt, M: m, K: k}, nil
}

// loadQTable loads a [rows, cols] table. GGUF stores matrix dims as [cols, rows].
func loadQTable(gguf *GGUFFile, name string, rows, cols int) (QTable, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return QTable{}, err
	}
	if err := validateMatrixTensor(info, name, rows, cols); err != nil {
		return QTable{}, err
	}
	dt := int(info.Type)
	if qtablePackedSupported(dt, cols) {
		rowBytes, err := qtableRowBytes(info.Type, cols)
		if err != nil {
			return QTable{}, err
		}
		packed := make([]byte, len(data))
		copy(packed, data)
		return QTable{Packed: packed, Dtype: dt, Rows: rows, Cols: cols, RowBytes: rowBytes}, nil
	}
	f32, err := dequantToF32(data, info.Type, rows*cols)
	if err != nil {
		return QTable{}, err
	}
	return QTable{F32: f32, Dtype: dt, Rows: rows, Cols: cols, RowBytes: cols * 4}, nil
}

func validateMatrixTensor(info *GGUFTensorInfo, name string, m, k int) error {
	if info.NDims != 2 {
		return fmt.Errorf("%s: expected 2D matrix [%d,%d], got %dD", name, m, k, info.NDims)
	}
	if info.Dims[0] != uint64(k) || info.Dims[1] != uint64(m) {
		return fmt.Errorf("%s: GGUF dims=[%d,%d], expected [k=%d,m=%d] for matrix [%d,%d]",
			name, info.Dims[0], info.Dims[1], k, m, m, k)
	}
	return nil
}

// LlamaState holds runtime buffers + KV cache.
type LlamaState struct {
	X      []float32 // hidden state [dim]
	XB     []float32 // post-norm scratch [dim]
	XB2    []float32 // attention output scratch [n_heads*head_dim]
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

// LoadLlamaModel builds a LlamaModel from a parsed GGUF file. Layer matrices
// and vocab tables are kept packed when their dtype has a runtime kernel; norms
// and tiny vectors are dequantized to f32.
func LoadLlamaModel(gguf *GGUFFile) (*LlamaModel, error) {
	m := gguf.Meta

	cfg := LlamaConfig{
		Architecture:   m.Architecture,
		BaseModelLabel: deriveBaseModelLabel(m),
		NumLayers:      m.NumLayers,
		EmbedDim:       m.EmbedDim,
		NumHeads:       m.NumHeads,
		NumKVHeads:     m.NumKVHeads,
		HeadDim:        m.HeadDim,
		VocabSize:      m.VocabSize,
		SeqLen:         m.SeqLen,
		IntermSize:     m.IntermSize,
		RMSNormEps:     m.RMSNormEps,
		RopeTheta:      m.RopeTheta,
	}
	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	}
	if m.KeyHeadDim > 0 && m.ValueHeadDim > 0 && m.KeyHeadDim != m.ValueHeadDim {
		return nil, fmt.Errorf("unsupported attention head dims: key=%d value=%d", m.KeyHeadDim, m.ValueHeadDim)
	}

	cfg.QKPermuted = (cfg.Architecture == "llama")

	// Cap context to keep KV cache reasonable on small machines.
	if cfg.SeqLen > 2048 {
		fmt.Printf("[tongue/model] capping seq_len from %d to 2048\n", cfg.SeqLen)
		cfg.SeqLen = 2048
	}

	w, err := loadWeights(gguf, &cfg)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Drop the raw GGUF byte buffer — layer weights and vocab tables have been
	// copied out, and norms are f32, so the original quantized blob can go.
	gguf.TensorData = nil
	runtime.GC()

	state := allocState(&cfg)
	precomputeRoPE(&state, &cfg)

	hasBias := w.Layers[0].BQ != nil
	hasQKNorm := w.Layers[0].QNorm != nil || w.Layers[0].KNorm != nil
	fmt.Printf("[tongue/model] loaded: arch=%s base=%s layers=%d dim=%d heads=%d kv_heads=%d head_dim=%d attn_dim=%d vocab=%d bias=%v qk_norm=%v qk_permuted=%v\n",
		cfg.Architecture, cfg.BaseModelLabel, cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads,
		cfg.HeadDim, cfg.NumHeads*cfg.HeadDim, cfg.VocabSize, hasBias, hasQKNorm, cfg.QKPermuted)

	return &LlamaModel{Config: cfg, Weights: *w, State: state}, nil
}

func deriveBaseModelLabel(m GGUFMetadata) string {
	name := ""
	if v, ok := m.KV["general.name"]; ok {
		if s, ok := v.(string); ok {
			name = strings.ToLower(s)
		}
	}
	arch := strings.ToLower(m.Architecture)
	if arch == "qwen3" || strings.Contains(name, "qwen3") {
		if m.EmbedDim == 1024 && m.NumLayers == 28 {
			return "qwen3-0.6b-base"
		}
		return "qwen3-base"
	}
	if strings.Contains(name, "smollm2") || (m.EmbedDim == 960 && m.NumLayers == 32) {
		return "smollm2-360m"
	}
	return arch
}

// loadWeights resolves every tensor in the GGUF. Large matrices/tables stay
// packed when possible; norms and small vectors are dequantized to f32.
func loadWeights(gguf *GGUFFile, cfg *LlamaConfig) (*LlamaWeights, error) {
	w := &LlamaWeights{}

	var err error
	w.TokenEmbed, err = loadQTable(gguf, "token_embd.weight", cfg.VocabSize, cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}

	w.OutputNorm, err = getF32Tensor(gguf, "output_norm.weight", cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("output_norm.weight: %w", err)
	}

	// Output (LM head) — may be tied to token embedding.
	if outInfo, ok := gguf.Tensors["output.weight"]; ok {
		fmt.Printf("[tongue/model] output.weight: type=%d\n", outInfo.Type)
		w.Output, err = loadQTable(gguf, "output.weight", cfg.VocabSize, cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("output.weight: %w", err)
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

		if l.WQ, err = loadQW(gguf, prefix+"attn_q.weight", qDim, dim); err != nil {
			return nil, fmt.Errorf("layer %d attn_q: %w", i, err)
		}
		if l.WK, err = loadQW(gguf, prefix+"attn_k.weight", kvDim, dim); err != nil {
			return nil, fmt.Errorf("layer %d attn_k: %w", i, err)
		}
		if l.WV, err = loadQW(gguf, prefix+"attn_v.weight", kvDim, dim); err != nil {
			return nil, fmt.Errorf("layer %d attn_v: %w", i, err)
		}
		if l.WO, err = loadQW(gguf, prefix+"attn_output.weight", dim, qDim); err != nil {
			return nil, fmt.Errorf("layer %d attn_output: %w", i, err)
		}

		l.BQ, _ = getF32TensorOptional(gguf, prefix+"attn_q.bias", qDim)
		l.BK, _ = getF32TensorOptional(gguf, prefix+"attn_k.bias", kvDim)
		l.BV, _ = getF32TensorOptional(gguf, prefix+"attn_v.bias", kvDim)
		l.BO, _ = getF32TensorOptional(gguf, prefix+"attn_output.bias", dim)

		if l.QNorm, err = getF32TensorOptional(gguf, prefix+"attn_q_norm.weight", cfg.HeadDim); err != nil {
			return nil, fmt.Errorf("layer %d attn_q_norm: %w", i, err)
		}
		if l.KNorm, err = getF32TensorOptional(gguf, prefix+"attn_k_norm.weight", cfg.HeadDim); err != nil {
			return nil, fmt.Errorf("layer %d attn_k_norm: %w", i, err)
		}

		if l.WGate, err = loadQW(gguf, prefix+"ffn_gate.weight", interm, dim); err != nil {
			return nil, fmt.Errorf("layer %d ffn_gate: %w", i, err)
		}
		if l.WUp, err = loadQW(gguf, prefix+"ffn_up.weight", interm, dim); err != nil {
			return nil, fmt.Errorf("layer %d ffn_up: %w", i, err)
		}
		if l.WDown, err = loadQW(gguf, prefix+"ffn_down.weight", dim, interm); err != nil {
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
		XB2:        make([]float32, cfg.NumHeads*cfg.HeadDim),
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

	// Token embedding lookup — dequantize only the selected row when packed.
	w.TokenEmbed.lookup(s.X, token)

	attnScale := float32(1.0 / math.Sqrt(float64(hd)))

	for layer := 0; layer < cfg.NumLayers; layer++ {
		l := &w.Layers[layer]

		// Attention pre-norm
		RMSNormInto(s.XB, s.X, l.AttnNorm, cfg.RMSNormEps)

		// Q, K, V projections — packed matvec (notorch nt_qmatvec, weights stay packed)
		l.WQ.matvec(s.Q, s.XB)
		l.WK.matvec(s.K, s.XB)
		l.WV.matvec(s.V, s.XB)

		addBias(s.Q, l.BQ)
		addBias(s.K, l.BK)
		addBias(s.V, l.BV)

		if cfg.QKPermuted {
			unpermuteQK(s.Q, cfg.NumHeads, hd)
			unpermuteQK(s.K, cfg.NumKVHeads, hd)
		}
		if l.QNorm != nil {
			RMSNormHeads(s.Q, l.QNorm, cfg.NumHeads, hd, cfg.RMSNormEps)
		}
		if l.KNorm != nil {
			RMSNormHeads(s.K, l.KNorm, cfg.NumKVHeads, hd, cfg.RMSNormEps)
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
		l.WO.matvec(s.XB, s.XB2)
		addBias(s.XB, l.BO)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}

		// MLP pre-norm
		RMSNormInto(s.XB, s.X, l.FFNNorm, cfg.RMSNormEps)

		// SwiGLU: silu(gate(x)) * up(x), then down(...)
		l.WGate.matvec(s.HB, s.XB)
		l.WUp.matvec(s.HB2, s.XB)
		for i := 0; i < cfg.IntermSize; i++ {
			s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
		}
		l.WDown.matvec(s.XB, s.HB)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	// Final norm + LM head
	RMSNorm(s.X, w.OutputNorm, cfg.RMSNormEps)
	w.Output.matvec(s.Logits, s.X)
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
