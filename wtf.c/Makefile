CC = gcc
CFLAGS = -O3 -march=native -Wall -Wextra
LDFLAGS = -lm

SRC = src/wtf.c
BIN = wtf

all: $(BIN)

$(BIN): $(SRC)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

debug: $(SRC)
	$(CC) -g -O0 -Wall -Wextra -fsanitize=address -o wtf_debug $< $(LDFLAGS)

clean:
	rm -f $(BIN) wtf_debug

.PHONY: all debug clean
