/**
 * On-GPU proof for `NdView!(T, N)`: `c = a + b` where BOTH inputs are
 * non-contiguous 2-D views of contiguous allocations.
 *
 * A contiguous test would pass even if strides were ignored entirely, so it
 * would prove nothing. Here:
 *
 *   a — TRANSPOSED view. Backing buffer is a dense COLS x ROWS matrix;
 *       the view is ROWS x COLS with strides [1, ROWS]. Innermost stride is
 *       ROWS, not 1.
 *   b — PADDED-ROW view, i.e. exactly the layout `cuMemAllocPitch` produces.
 *       Backing buffer has PAD elements per row (PAD * float.sizeof == 512 B,
 *       a realistic driver pitch) of which only COLS are live; the view is
 *       ROWS x COLS with strides [PAD, 1]. The padding is filled with a
 *       poison value so that ignoring the row stride is loudly wrong.
 *   c — dense output, strides [COLS, 1].
 *
 * With a[i,j] = 1000*j + i and b[i,j] = -(2*i + j) the expected result is the
 * clean closed form  c[i,j] == 999*j - i,  which is hand-checkable.
 */
module dcompute.tests.ndview;

import std.exception : enforce;
import std.conv : to;
import std.stdio;

import dcompute.driver.cuda;
import dcompute.std.ndview;
import dcompute.tests.dummykernels : ndAddStrided2;

private enum size_t ROWS = 64;
private enum size_t COLS = 48;
private enum size_t PAD  = 128;          // 128 * 4 B == 512 B, a real pitch
private enum float  POISON = 999_999.0f;

/// Point an NdView at a device allocation. `Buffer.raw` is a CUdeviceptr;
/// `GlobalPointer!T` is a plain `T*` wrapper, so this is a reinterpret, and
/// the resulting pointer is only ever dereferenced on the device.
private NdView!(float,2) viewOf(ref Buffer!float b, size_t[2] shape, size_t[2] strides)
{
    NdView!(float,2) v;
    v.ptr.ptr = cast(float*) b.raw;
    v.shape   = shape;
    v.strides = strides;
    return v;
}

/// Host-side checks of the indexing math. `NdView` is
/// @compute(hostAndDevice), so this is literally the same code the device runs.
private void runHostIndexTests()
{
    float[24] data;
    foreach (i; 0 .. 24) data[i] = i;

    // 4x6 dense
    NdView!(float,2) dense;
    dense.ptr.ptr = data.ptr;
    dense.shape   = [4, 6];
    dense.strides = rowMajorStrides!2([4, 6]);
    enforce(dense.strides == [6UL, 1UL], "rowMajorStrides");
    enforce(dense.length == 24, "length");
    enforce(dense.offsetOf([2, 3]) == 15, "offsetOf dense");
    enforce(dense[2, 3] == 15.0f, "opIndex dense");
    enforce(dense[[2, 3]] == 15.0f, "opIndex array form");
    enforce(dense.offsetOfLinear(15) == 15, "offsetOfLinear dense == identity");
    dense[1, 1] = -1.0f;
    enforce(data[7] == -1.0f, "opIndexAssign");
    dense.atAssign(7, 42.0f);
    enforce(data[7] == 42.0f, "atAssign");
    data[7] = 7.0f;

    // 6x4 transposed view of the same memory
    NdView!(float,2) tr;
    tr.ptr.ptr = data.ptr;
    tr.shape   = [6, 4];
    tr.strides = [1, 6];
    enforce(tr.offsetOf([3, 2]) == 3 + 2 * 6, "offsetOf transposed");
    enforce(tr[3, 2] == 15.0f, "opIndex transposed");
    // linear id 14 -> (row 3, col 2) -> offset 15, i.e. NOT 14.
    enforce(tr.offsetOfLinear(14) == 15, "offsetOfLinear transposed");

    // stride 0 => broadcast (design headroom only; no rules implemented)
    NdView!(float,2) bc;
    bc.ptr.ptr = data.ptr;
    bc.shape   = [4, 6];
    bc.strides = [0, 1];
    enforce(bc[0, 5] == bc[3, 5], "stride-0 aliases");

    // 1-D still works
    NdView!(float,1) v1;
    v1.ptr.ptr = data.ptr;
    v1.shape   = [24];
    v1.strides = [2];
    enforce(v1[5] == 10.0f, "1-D strided opIndex");
    enforce(v1.at(5) == 10.0f, "1-D strided at");
}

/// Launch the strided add on the GPU and validate against a host reference.
void runNdViewTests()
{
    ensureInit();
    runHostIndexTests();

    // ---- host data ----------------------------------------------------
    auto aSrc = new float[COLS * ROWS];       // dense COLS x ROWS
    foreach (p; 0 .. COLS)
        foreach (q; 0 .. ROWS)
            aSrc[p * ROWS + q] = p * 1000.0f + q;

    auto bSrc = new float[ROWS * PAD];        // ROWS rows of PAD, COLS live
    foreach (i; 0 .. ROWS)
        foreach (j; 0 .. PAD)
            bSrc[i * PAD + j] = (j < COLS) ? -(2.0f * i + j) : POISON;

    auto cDst = new float[ROWS * COLS];
    cDst[] = float.nan;

    // ---- device ---------------------------------------------------------
    auto bufA = Buffer!float(aSrc); scope(exit) bufA.release();
    auto bufB = Buffer!float(bSrc); scope(exit) bufB.release();
    auto bufC = Buffer!float(cDst); scope(exit) bufC.release();
    bufA.copy!(Copy.hostToDevice);
    bufB.copy!(Copy.hostToDevice);

    auto a = viewOf(bufA, [ROWS, COLS], [1, ROWS]);              // transposed
    auto b = viewOf(bufB, [ROWS, COLS], [PAD, 1]);               // pitched row
    auto c = viewOf(bufC, [ROWS, COLS], rowMajorStrides!2([ROWS, COLS]));

    enum uint block = 256;
    immutable uint grid = cast(uint)((ROWS * COLS + block - 1) / block);
    launch!ndAddStrided2([grid,1,1],[block,1,1], c, a, b);
    bufC.copy!(Copy.deviceToHost);

    // ---- validate against the host reference ---------------------------
    size_t checked;
    foreach (i; 0 .. ROWS)
        foreach (j; 0 .. COLS)
        {
            immutable float want = aSrc[j * ROWS + i] + bSrc[i * PAD + j];
            immutable float got  = cDst[i * COLS + j];
            enforce(got == want,
                    "NdView strided add mismatch at [" ~ i.to!string ~ "," ~
                    j.to!string ~ "]: got " ~ got.to!string ~ " want " ~
                    want.to!string);
            // the closed form, as a cross-check on the test data itself
            enforce(want == 999.0f * j - i, "reference formula");
            ++checked;
        }

    // ---- prove the test would FAIL if strides were ignored --------------
    // "Strides ignored" means offset == linear index, i.e. reading aSrc and
    // bSrc densely in row-major order. Show that differs from the truth.
    {
        immutable size_t lin = 1;              // element [0,1]
        immutable float naive = aSrc[lin] + bSrc[lin];
        immutable float truth = cDst[lin];
        enforce(naive != truth,
                "test is not stride-sensitive — this must never happen");
        writefln("  stride-sensitivity: element [0,1] is %s; ignoring strides " ~
                 "would give %s", truth, naive);
    }

    writefln("  c[0,0]=%s c[0,1]=%s c[1,0]=%s c[63,47]=%s (== 999*j - i)",
             cDst[0], cDst[1], cDst[COLS], cDst[ROWS * COLS - 1]);
    writefln("NdView strided add: %s/%s elements match the host reference. PASSED.",
             checked, ROWS * COLS);
}
