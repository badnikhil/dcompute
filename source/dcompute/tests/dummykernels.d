@compute(CompileFor.deviceOnly)
module dcompute.tests.dummykernels;
pragma(LDC_no_moduleinfo);

import ldc.dcompute;
import dcompute.std.index;

@kernel() void saxpy(GlobalPointer!(float) res,
                   float alpha,GlobalPointer!(float) x,
                   GlobalPointer!(float) y, 
                   size_t N)
{
    auto i = GlobalIndex.x;
    if (i >= N) return;
    res[i] = alpha*x[i] + y[i];
}

alias aagf = AutoIndexed!(GlobalPointer!(float));

@kernel() void auto_index_test(aagf a,
                             aagf b,
                             aagf c)
{
    a = b + c;
}

import dcompute.std.ndview;

/**
 * Strided elementwise add over N-d views: `c = a + b`.
 *
 * The whole point of the test is that `a`, `b` and `c` may have completely
 * different strides — the kernel never assumes contiguity. Both indexing
 * paths are exercised deliberately:
 *   - `a.at(i)`  : linear id -> multi-index -> offset (the hot path),
 *   - `b[r, col]`: explicit multi-index (opIndex),
 * and the write goes through `opIndex` too. If either path is wrong, or if
 * strides were ignored, the result differs from the host reference.
 *
 * `NdView` is taken BY VALUE, which is the ABI question this test answers.
 */
@kernel() void ndAddStrided2(NdView!(float,2) c,
                             NdView!(float,2) a,
                             NdView!(float,2) b)
{
    size_t i = GlobalIndex.x;
    if (i >= c.length) return;

    size_t row = i / c.shape[1];
    size_t col = i % c.shape[1];

    c[row, col] = a.at(i) + b[row, col];
}
