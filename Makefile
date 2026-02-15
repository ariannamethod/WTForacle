# WTForacle Makefile — Go engine (Qwen2.5 0.5B GGUF)

UNAME := $(shell uname)
ifeq ($(UNAME),Darwin)
  EXT = dylib
else
  EXT = so
endif

# Build Go shared library
wtf-lib:
	cd wtf && go build -buildmode=c-shared -o ../libwtf.$(EXT) .

# Download Qwen2.5 0.5B weights from HuggingFace (v2, Q4_0, ~352MB)
wtf-weights:
	mkdir -p wtfweights
	curl -L -o wtfweights/wtf_qwen_v2_q4_0.gguf \
	  https://huggingface.co/ataeff/WTForacle/resolve/main/wtf_q/wtf_qwen_v2_q4_0.gguf

# Run
run: wtforacle.py
	python3 wtforacle.py

clean:
	rm -f libwtf.dylib libwtf.so libwtf.h

.PHONY: wtf-lib wtf-weights run clean
