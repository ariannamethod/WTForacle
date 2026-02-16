# WTForacle Makefile — Go engine (SmolLM2 360M GGUF)

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  EXT = dylib
else
  EXT = so
endif

# Build Go shared library
wtf-lib:
	cd wtf && go build -buildmode=c-shared -o ../libwtf.$(EXT) .

# Download SmolLM2 360M weights from HuggingFace (v2, Q4_0, ~229MB)
wtf-weights:
	mkdir -p wtfweights
	curl -L -o wtfweights/wtf360_v2_q4_0.gguf \
	  https://huggingface.co/ataeff/WTForacle/resolve/main/ws360/wtf360_v2_q4_0.gguf

# Build + Run
run: wtf-lib
	python3 wtforacle.py

clean:
	rm -f libwtf.dylib libwtf.so libwtf.h

.PHONY: wtf-lib wtf-weights run clean
