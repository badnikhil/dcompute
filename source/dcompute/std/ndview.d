/**
 * `NdView!(T, N)` — an N-dimensional strided view over device memory.
 *
 * This is the host/device shared contract for shaped data: a POD triple of
 * (pointer, shape, strides) that can be passed **by value** as a `@kernel`
 * parameter and indexed from device code.
 *
 * ---------------------------------------------------------------------------
 * Conventions (locked — downstream packages depend on these)
 * ---------------------------------------------------------------------------
 *
 * $(UL
 * $(LI $(B Strides are in ELEMENTS, never bytes.) The element offset of a
 *      multi-index `idx` is `sum(idx[d] * strides[d])`, and the element is
 *      `ptr[offset]` — pointer arithmetic on `T*`, so a byte stride would be
 *      wrong by a factor of `T.sizeof`.)
 * $(LI $(B Index order is row-major-ish:) dimension `0` is the outermost /
 *      slowest-varying, dimension `N-1` the innermost / fastest-varying. A
 *      dense C-order array has `strides[N-1] == 1`. This only fixes the
 *      meaning of the *linear* traversal order; the strides themselves are
 *      free, so a transposed or column-major view is just a permutation of
 *      `shape`/`strides`.)
 * $(LI $(B `strides[d] == 0` is reserved for broadcasting) — a zero stride
 *      makes every index along `d` alias the same element. `offsetOf` already
 *      does the right thing for it; the *rules* for deriving broadcast shapes
 *      are deliberately not implemented here (they belong to the framework
 *      layer).)
 * $(LI $(B No ownership.) `NdView` never allocates or frees. It is a view;
 *      the lifetime of `ptr` belongs to whatever allocated it (`Buffer`,
 *      `PitchedBuffer`, a pool).)
 * $(LI $(B No `ref` returns.) Element accessors return/take values
 *      (`opIndex`/`opIndexAssign`), never `ref T`. This is not a style
 *      choice: LDC 1.43.0 miscompiles a `ref T` return whose referent is
 *      reached through a `Pointer!(AddrSpace.Global, T)` field — the callee
 *      emits an extra `ld.global.b64` at the element address instead of
 *      returning it, so a `float` view faults with
 *      `CUDA_ERROR_MISALIGNED_ADDRESS` (an 8-byte load at a 4-byte-aligned
 *      address). Verified on LDC 1.43.0 in both debug and release builds; see
 *      the PTX quoted in the pull request. Passing the element by value emits
 *      the correct `ld.global.b32`/`st.global.b32`.)
 * )
 *
 * ---------------------------------------------------------------------------
 * Mapping a pitched allocation onto element strides
 * ---------------------------------------------------------------------------
 *
 * `cuMemAllocPitch` returns a $(B byte) pitch: the byte distance between the
 * starts of consecutive rows, rounded up by the driver for coalescing (512 on
 * current hardware). `NdView` strides are elements, so for a 2-D pitched
 * allocation of `height` rows by `width` elements:
 *
 * ---
 * assert(pitchBytes % T.sizeof == 0);
 * view.shape   = [height, width];
 * view.strides = [pitchBytes / T.sizeof, 1];
 * ---
 *
 * The divide is exact in practice — `cuMemAllocPitch` pitches are multiples of
 * the texture alignment (512) and `T.sizeof` is a power of two ≤ 16 — but it
 * is a genuine precondition, so assert it rather than assume it. Forgetting
 * the division is the classic pitched-memory bug: it silently reads
 * `T.sizeof` times too far and usually still "works" for `T == ubyte`.
 *
 * For a 3-D pitched allocation with `depth` slices of `height` rows:
 * `strides = [pitchBytes / T.sizeof * height, pitchBytes / T.sizeof, 1]`.
 *
 * ---------------------------------------------------------------------------
 * Why this module is `@compute(CompileFor.hostAndDevice)`
 * ---------------------------------------------------------------------------
 *
 * LDC's `gen/semantic-dcompute.cpp` rejects any call from `@compute` code into
 * a host-only module ("can only call functions from other `@compute` modules
 * in `@compute` code"). Kernels call `opIndex`/`offsetOf`, so they must live
 * in a `@compute` module; `hostAndDevice` makes the same code usable to build
 * views on the host. Consequently this module must stay POD and Phobos-free:
 * no GC, no exceptions, no imports beyond `ldc.dcompute`. Everything is
 * templated, so nothing is codegenned for the device unless a kernel
 * instantiates it.
 */
@compute(CompileFor.hostAndDevice) module dcompute.std.ndview;

import ldc.dcompute;

/**
 * A strided N-dimensional view over global device memory.
 *
 * Params:
 *   T = element type
 *   N = rank (number of dimensions), `>= 1`
 *
 * Layout is `{ T* ; size_t[N] ; size_t[N] }`, identical on host and device
 * (both 64-bit), which is what lets it be passed by value through the kernel
 * ABI. The host builds one by pointing `ptr.ptr` at a `CUdeviceptr`.
 */
struct NdView(T, size_t N)
if (N >= 1)
{
    /// Base of the view. Element `i` of the underlying allocation is `ptr[i]`.
    GlobalPointer!T ptr;

    /// Extent of each dimension. `shape[0]` is outermost, `shape[N-1]` innermost.
    size_t[N] shape;

    /// Distance to the next element along each dimension, in ELEMENTS.
    /// `0` means "broadcast": every index along that dimension aliases one element.
    size_t[N] strides;

    /// Rank, as a compile-time constant.
    enum size_t rank = N;

    /// Element type, for generic code.
    alias ElementType = T;

    /// Number of logical elements, `shape[0] * ... * shape[N-1]`.
    /// Note this counts broadcast elements too; it is the size of the
    /// iteration space, not of the allocation.
    @property size_t length()
    {
        size_t n = 1;
        static foreach (d; 0 .. N)
            n *= shape[d];
        return n;
    }

    /**
     * Element offset (in elements, from `ptr`) of a multi-index.
     * Unrolled over `N` at compile time: `N` multiply-adds, no loop, no
     * division.
     */
    size_t offsetOf(size_t[N] idx)
    {
        size_t o = 0;
        static foreach (d; 0 .. N)
            o += idx[d] * strides[d];
        return o;
    }

    /**
     * Element at a multi-index: `v[[i, j]]`, or `v[i, j]` (and `v[i]` when
     * `N == 1`) via the variadic overload.
     *
     * Returned BY VALUE, deliberately — see the "no `ref` returns" note in
     * the module documentation. Write with `v[i, j] = x` (`opIndexAssign`).
     */
    T opIndex(size_t[N] idx)
    {
        return ptr[offsetOf(idx)];
    }

    /// ditto
    T opIndex(Idx...)(Idx idx)
    if (Idx.length == N)
    {
        size_t[N] a;
        static foreach (d; 0 .. N)
            a[d] = idx[d];
        return ptr[offsetOf(a)];
    }

    /// Store `value` at a multi-index: `v[[i, j]] = value` / `v[i, j] = value`.
    void opIndexAssign(T value, size_t[N] idx)
    {
        ptr[offsetOf(idx)] = value;
    }

    /// ditto
    void opIndexAssign(Idx...)(T value, Idx idx)
    if (Idx.length == N)
    {
        size_t[N] a;
        static foreach (d; 0 .. N)
            a[d] = idx[d];
        ptr[offsetOf(a)] = value;
    }

    /**
     * Element offset of the `i`th element in row-major logical order.
     *
     * This is the kernel hot path: one work item takes its global linear id,
     * decomposes it into a multi-index and folds that against the strides.
     * The loop is unrolled over `N` at compile time, and the outermost
     * dimension needs no div/mod (its quotient is whatever is left), so the
     * cost is `N-1` divmods and `N` multiply-adds.
     *
     * `i` must be `< length`; the caller does the bounds check (kernels have
     * to do one anyway to handle a partial final block).
     */
    size_t offsetOfLinear(size_t i)
    {
        size_t o = 0;
        static foreach_reverse (d; 1 .. N)
        {
            o += (i % shape[d]) * strides[d];
            i /= shape[d];
        }
        o += i * strides[0];
        return o;
    }

    /// Element at linear index `i` in row-major logical order (by value).
    T at(size_t i)
    {
        return ptr[offsetOfLinear(i)];
    }

    /// Store `value` at linear index `i` in row-major logical order.
    void atAssign(size_t i, T value)
    {
        ptr[offsetOfLinear(i)] = value;
    }
}

/// Dense row-major (C-order) strides for `shape`: innermost dimension has stride 1.
size_t[N] rowMajorStrides(size_t N)(size_t[N] shape)
{
    size_t[N] s;
    size_t acc = 1;
    static foreach_reverse (d; 0 .. N)
    {
        s[d] = acc;
        acc *= shape[d];
    }
    return s;
}
