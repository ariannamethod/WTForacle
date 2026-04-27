// cbridge.c — single-TU bridge so cgo picks up the vendored ariannamethod/
// sources without symlinks or external archives. Real code lives in
// ../ariannamethod/{wtf_kernels.c,notorch.c}; this file just pulls them
// into the wtf cgo build.

#include "../ariannamethod/wtf_kernels.c"
#include "../ariannamethod/notorch.c"
