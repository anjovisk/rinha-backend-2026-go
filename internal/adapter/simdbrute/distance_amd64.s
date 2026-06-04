#include "textflag.h"

// func l2sq14(a, b *float32) float32
//
// Computes sum((a[i]-b[i])^2) for i in [0,14) using AVX2.
//
// Layout of 14 float32 (56 bytes):
//   bytes  0–31  → a[0..7]   processed with 256-bit YMM
//   bytes 32–47  → a[8..11]  processed with 128-bit XMM
//   bytes 48–51  → a[12]     scalar
//   bytes 52–55  → a[13]     scalar
//
// Stack frame: 0 local bytes; args: a(8) + b(8) + ret(4) = 20 bytes.
TEXT ·l2sq14(SB), NOSPLIT, $0-20
    MOVQ a+0(FP), SI        // SI = a
    MOVQ b+8(FP), DI        // DI = b

    // ── a[0..7] × b[0..7]: 256-bit path ─────────────────────────────────────
    VMOVUPS (SI), Y1         // Y1 = a[0..7]
    VMOVUPS (DI), Y2         // Y2 = b[0..7]
    VSUBPS  Y2, Y1, Y1      // Y1 = a[0..7] - b[0..7]
    VMULPS  Y1, Y1, Y0      // Y0 = (a-b)^2 [0..7]

    // ── a[8..11] × b[8..11]: 128-bit path ────────────────────────────────────
    VMOVUPS 32(SI), X1       // X1 = a[8..11]
    VMOVUPS 32(DI), X2       // X2 = b[8..11]
    VSUBPS  X2, X1, X1      // X1 = a[8..11] - b[8..11]
    VMULPS  X1, X1, X1      // X1 = (a-b)^2 [8..11]

    // ── Reduce Y0 (8 floats) → X0 (4 floats) then add X1 ────────────────────
    VEXTRACTF128 $1, Y0, X2  // X2 = Y0[4..7]
    VADDPS X2, X0, X0        // X0 = Y0[0..3] + Y0[4..7]
    VADDPS X1, X0, X0        // X0 += (a-b)^2 [8..11]

    // ── Horizontal sum X0 (4 floats → 1 float) ───────────────────────────────
    VHADDPS X0, X0, X0       // X0 = [s01, s23, s01, s23]
    VHADDPS X0, X0, X0       // X0 = [total, ...]

    // ── a[12] × b[12]: scalar ────────────────────────────────────────────────
    MOVSS   48(SI), X1
    MOVSS   48(DI), X2
    SUBSS   X2, X1           // X1 = a[12] - b[12]
    MULSS   X1, X1           // X1 = (a[12]-b[12])^2
    ADDSS   X1, X0

    // ── a[13] × b[13]: scalar ────────────────────────────────────────────────
    MOVSS   52(SI), X1
    MOVSS   52(DI), X2
    SUBSS   X2, X1           // X1 = a[13] - b[13]
    MULSS   X1, X1           // X1 = (a[13]-b[13])^2
    ADDSS   X1, X0

    VZEROUPPER               // avoid AVX-SSE transition penalty
    MOVSS   X0, ret+16(FP)
    RET
