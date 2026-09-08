module dcompute.driver.cuda.buffer;

import dcompute.driver.cuda;

struct Buffer(T)
{
    size_t raw;

	// Host memory associated with this buffer
    T[] hostMemory;

    // Shared atomic reference count for the owned CUdeviceptr (see rcCreate/
    // rcRetain/rcRelease in dcompute.driver.cuda). null when this Buffer does
    // not own a handle (e.g. a non-owning view created via
    // UnifiedBuffer.asBuffer). Copies share the same counter so the underlying
    // device allocation is freed exactly once, when the last owner dies.
    private shared(int)* _rc;

    this(size_t elems)
    {
        status = cast(Status)cuMemAlloc(&raw,elems * T.sizeof);
        checkErrors();
        hostMemory = null;
        if (raw != 0)
            _rc = rcCreate();
    }

    this(T[] arr)
    {
        status = cast(Status)cuMemAlloc(&raw,arr.length * T.sizeof);
        checkErrors();
        hostMemory = arr;
        if (raw != 0)
            _rc = rcCreate();
    }

    // Copies bump the shared refcount so the handle survives until the last
    // copy is destroyed.
    this(this)
    {
        rcRetain(_rc);
    }

    // Last owner frees the device allocation. The CUresult is deliberately
    // ignored: throwing from a destructor (which may run as a GC finalizer, on
    // a thread without a current CUDA context, or after driver teardown at
    // program exit) is unsafe — in those cases cuMemFree simply reports an
    // error we cannot act on anyway.
    ~this()
    {
        if (rcRelease(_rc) && raw != 0)
            cuMemFree(raw);
        raw = 0;
        // hostMemory is a GC-owned slice: never free it, just drop the reference.
        hostMemory = null;
    }
    void copy(Copy c)()
    {
        static if (c == Copy.hostToDevice)
        {
            status = cast(Status)cuMemcpyHtoD(raw, hostMemory.ptr,hostMemory.length * T.sizeof);
        }
        else static if  (c == Copy.deviceToHost)
        {
            status = cast(Status)cuMemcpyDtoH(hostMemory.ptr,raw,hostMemory.length * T.sizeof);
        }
        checkErrors();
    }
    /// Fill the buffer with `value` on the device (cuMemsetD8/D16/D32,
    /// chosen by T.sizeof). Fills the whole allocation backing `raw`;
    /// the element count is queried from the driver so both constructors
    /// are supported.
    void memset(T value)
    {
        size_t base, nbytes;
        status = cast(Status)cuMemGetAddressRange(&base,&nbytes,raw);
        checkErrors();
        static if (T.sizeof == 1)
            status = cast(Status)cuMemsetD8(raw, *cast(ubyte*)&value, nbytes);
        else static if (T.sizeof == 2)
            status = cast(Status)cuMemsetD16(raw, *cast(ushort*)&value, nbytes / 2);
        else static if (T.sizeof == 4)
            status = cast(Status)cuMemsetD32(raw, *cast(uint*)&value, nbytes / 4);
        else
            static assert(false, "memset requires a 1-, 2- or 4-byte element type");
        checkErrors();
    }
    alias hostArgOf(U : GlobalPointer!T) = raw;
    void release()
    {
        if (_rc !is null)
        {
            // Owning buffer: drop this copy's ownership. Only the LAST owner
            // actually frees — with other copies still alive the free is
            // deferred to the last copy's release()/~this(), so the surviving
            // copies keep a valid handle and nothing is freed twice.
            if (rcRelease(_rc) && raw != 0)
            {
                status = cast(Status)cuMemFree(raw);
                checkErrors();
            }
        }
        else if (raw != 0)
        {
            // Non-owning / manually assembled buffer: legacy behaviour, free
            // immediately (the caller manages the lifetime by hand).
            status = cast(Status)cuMemFree(raw);
            checkErrors();
        }
        raw = 0;
        hostMemory = null;
        // Handle and ownership are both gone; ~this is now a guaranteed no-op.
    }
}

/**
 * 2D device buffer backed by a pitched allocation (cuMemAllocPitch).
 *
 * Rows are `width` elements wide; row `y` starts at `raw + y * pitch`,
 * where `pitch >= width * T.sizeof` is chosen by the driver to satisfy the
 * device's alignment/coalescing requirements.
 *
 * `hostMemory` is a dense row-major slice (length == width * height);
 * copy!() converts between the dense host layout and the pitched device
 * layout with a single cuMemcpy2D.
 *
 * Cleanup is manual via release(), matching Buffer.
 */
struct PitchedBuffer(T)
{
    size_t raw;    // CUdeviceptr of the pitched allocation
    size_t pitch;  // byte offset between the starts of consecutive rows
    size_t width;  // elements per row
    size_t height; // number of rows

	// Dense row-major host memory associated with this buffer
    T[] hostMemory;

    this(size_t width, size_t height)
    {
        // cuMemAllocPitch only accepts 4, 8 or 16 as the element size hint;
        // pass T.sizeof when it is one of those, otherwise fall back to 4.
        enum uint elementSize =
            (T.sizeof == 8 || T.sizeof == 16) ? cast(uint)T.sizeof : 4u;
        this.width  = width;
        this.height = height;
        status = cast(Status)cuMemAllocPitch(&raw,&pitch,width * T.sizeof,
                                             height,elementSize);
        checkErrors();
        hostMemory = null;
    }

    this(T[] arr, size_t width)
    {
        assert(width > 0 && arr.length % width == 0,
               "array length must be a multiple of the row width");
        this(width, arr.length / width);
        hostMemory = arr;
    }

    void copy(Copy c)()
    {
        CUDA_MEMCPY2D desc; // zero-initialised; unused fields must stay 0/null
        desc.WidthInBytes = width * T.sizeof;
        desc.Height       = height;
        static if (c == Copy.hostToDevice)
        {
            desc.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
            desc.srcHost       = hostMemory.ptr;
            desc.srcPitch      = width * T.sizeof;
            desc.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
            desc.dstDevice     = raw;
            desc.dstPitch      = pitch;
        }
        else static if (c == Copy.deviceToHost)
        {
            desc.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
            desc.srcDevice     = raw;
            desc.srcPitch      = pitch;
            desc.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
            desc.dstHost       = hostMemory.ptr;
            desc.dstPitch      = width * T.sizeof;
        }
        copy2D(desc);
    }

    /// Fill the `width` elements of every row with `value` on the device
    /// (cuMemsetD2D8/D2D16/D2D32, chosen by T.sizeof). Padding bytes
    /// between rows are left untouched.
    void memset(T value)
    {
        static if (T.sizeof == 1)
            status = cast(Status)cuMemsetD2D8(raw, pitch, *cast(ubyte*)&value, width, height);
        else static if (T.sizeof == 2)
            status = cast(Status)cuMemsetD2D16(raw, pitch, *cast(ushort*)&value, width, height);
        else static if (T.sizeof == 4)
            status = cast(Status)cuMemsetD2D32(raw, pitch, *cast(uint*)&value, width, height);
        else
            static assert(false, "memset requires a 1-, 2- or 4-byte element type");
        checkErrors();
    }

    void release()
    {
        status = cast(Status)cuMemFree(raw);
        checkErrors();
        raw = 0;
        pitch = 0;
        width = 0;
        height = 0;
        hostMemory = null;
    }
}

alias bf = Buffer!float;
