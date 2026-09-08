module dcompute.driver.cuda.memory;

import dcompute.driver.error;
import dcompute.driver.cuda;

// void pointer like
struct MemoryPointer
{
    size_t raw;
    static MemoryPointer allocate(size_t nbytes)
    {
        MemoryPointer ret;
        status = cast(Status)cuMemAlloc(&ret.raw,nbytes);
        checkErrors();
        return ret;
    }

    /// Allocate pitched (2D) memory: `height` rows of `widthInBytes` bytes.
    /// The driver chooses `pitch` (the byte offset between the starts of
    /// consecutive rows, >= widthInBytes) to satisfy the device's alignment
    /// requirements. `elementSizeBytes` must be 4, 8 or 16.
    static MemoryPointer allocatePitch(out size_t pitch, size_t widthInBytes,
                                       size_t height, uint elementSizeBytes)
    {
        MemoryPointer ret;
        status = cast(Status)cuMemAllocPitch(&ret.raw,&pitch,widthInBytes,
                                             height,elementSizeBytes);
        checkErrors();
        return ret;
    }

    Memory addressRange()
    {
        Memory ret;
        status = cast(Status)cuMemGetAddressRange(&ret.ptr.raw,&ret.length,raw);
        checkErrors();
        return ret;
    }

}

// void[] like
struct Memory
{
    MemoryPointer ptr;
    size_t length;

    enum CopySource
    {
        Host,
        Device,
        Array
    }

    // Typed 1D copies live on Buffer (cuMemcpyHtoD/DtoH); pitched 2D copies
    // live on PitchedBuffer. Raw descriptor-based copies: copy2D/copy3D below.

    /// Fill the memory range with a byte value (cuMemsetD8).
    void set(ubyte value)
    {
        status = cast(Status)cuMemsetD8(ptr.raw, value, length);
        checkErrors();
    }
}

/// Execute a 2D copy described by `desc` (cuMemcpy2D). Unused fields of the
/// descriptor must be left zero/null.
void copy2D(ref CUDA_MEMCPY2D desc)
{
    status = cast(Status)cuMemcpy2D(&desc);
    checkErrors();
}

/// Execute a 3D copy described by `desc` (cuMemcpy3D). Unused fields of the
/// descriptor must be left zero/null.
void copy3D(ref CUDA_MEMCPY3D desc)
{
    status = cast(Status)cuMemcpy3D(&desc);
    checkErrors();
}
