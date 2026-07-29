# WTForacle Makefile — single Go binary, BLAS via vendored notorch.

QWEN3_Q8_SHARED ?= /Users/ataeff/arianna-shared/wtforacle/quants/qwen3-0p6b-long-v1-step300/wtforacle_qwen3_0p6b_long_v1_step300_q8_0.gguf
QWEN3_Q8_LOCAL := wtfweights/wtforacle_qwen3_0p6b_long_v1_step300_q8_0.gguf

# Build the wtforacle binary (REPL by default, -prompt for one-shot).
wtforacle:
	go build -o wtforacle ./cmd/wtf/

# Download SmolLM2 360M weights (Q4_0, ~229MB) from HuggingFace.
wtf-weights:
	mkdir -p wtfweights
	curl -L -o wtfweights/wtf360_v2_q4_0.gguf \
	  https://huggingface.co/ataeff/WTForacle/resolve/main/ws360/wtf360_v2_q4_0.gguf

# Link the trusted local Qwen3-0.6B Q8_0 candidate from the shared workspace.
qwen3-weights-local:
	mkdir -p wtfweights
	test -f "$(QWEN3_Q8_SHARED)"
	ln -sf "$(QWEN3_Q8_SHARED)" "$(QWEN3_Q8_LOCAL)"

# Build + run the REPL.
run: wtforacle
	./wtforacle

run-qwen3: wtforacle qwen3-weights-local
	./wtforacle

clean:
	rm -f wtforacle

.PHONY: wtforacle wtf-weights qwen3-weights-local run run-qwen3 clean
