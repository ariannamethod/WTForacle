# WTForacle Makefile — Go engine (SmolLM2 360M GGUF)

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  EXT = dylib
else
  EXT = so
endif

# Build Go shared library
wtf-lib:
	go build -buildmode=c-shared -o libwtf.$(EXT) ./cmd/wtf-lib/

# Build native Go CLI
wtf-cli:
	go build -o wtf-cli ./cmd/wtf/

# Download SmolLM2 360M weights from HuggingFace (v2, Q4_0, ~229MB)
wtf-weights:
	mkdir -p wtfweights
	curl -L -o wtfweights/wtf360_v2_q4_0.gguf \
	  https://huggingface.co/ataeff/WTForacle/resolve/main/ws360/wtf360_v2_q4_0.gguf

# Build + Run
run: wtf-lib
	python3 wtforacle.py

clean:
	rm -f libwtf.dylib libwtf.so libwtf.h wtf-cli

.PHONY: wtf-lib wtf-cli wtf-weights run clean
