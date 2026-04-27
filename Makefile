# WTForacle Makefile — single Go binary, BLAS via vendored notorch.

# Build the wtforacle binary (REPL by default, -prompt for one-shot).
wtforacle:
	go build -o wtforacle ./cmd/wtf/

# Download SmolLM2 360M weights (Q4_0, ~229MB) from HuggingFace.
wtf-weights:
	mkdir -p wtfweights
	curl -L -o wtfweights/wtf360_v2_q4_0.gguf \
	  https://huggingface.co/ataeff/WTForacle/resolve/main/ws360/wtf360_v2_q4_0.gguf

# Build + run the REPL.
run: wtforacle
	./wtforacle

clean:
	rm -f wtforacle

.PHONY: wtforacle wtf-weights run clean
