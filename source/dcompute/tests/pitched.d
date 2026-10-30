module dcompute.tests.pitched;

version (DComputeTesting) {
    version = DComputeTestCUDA;
}

version (DComputeTestCUDA):

import std.stdio;
import std.exception : enforce;

import dcompute.driver.cuda;

// Pitched memory / 2D & 3D copies / device memset
void runPitchedTests()
{
    enum size_t W = 33; // deliberately awkward width so the driver must pad rows
    enum size_t H = 7;

    // Pitched allocation: pitch must cover a row and be aligned.
    auto pb = PitchedBuffer!float(W, H); scope(exit) pb.release();
    enforce(pb.pitch >= W * float.sizeof);
    enforce(pb.pitch % 16 == 0);
    writeln("\nPitchedBuffer!float(", W, "x", H, ") pitch = ", pb.pitch, " bytes");

    // Host->device->host round trip of a dense row-major pattern.
    float[W * H] pattern;
    foreach (i; 0 .. W * H) pattern[i] = i;
    auto pb2 = PitchedBuffer!float(pattern[], W); scope(exit) pb2.release();
    pb2.copy!(Copy.hostToDevice);

    // Row addressing must honour the pitch: read one row back directly
    // from raw + y*pitch and compare against the dense host row.
    float[W] row;
    status = cast(Status)cuMemcpyDtoH(row.ptr, pb2.raw + 3 * pb2.pitch, W * float.sizeof);
    checkErrors();
    foreach (i; 0 .. W) enforce(row[i] == pattern[3 * W + i]);

    // Full 2D device->host copy must reproduce the pattern.
    pattern[] = 0.0f;
    pb2.copy!(Copy.deviceToHost);
    foreach (i; 0 .. W * H) enforce(pattern[i] == i);

    // 2D memset (cuMemsetD2D32) fills every element of every row.
    pb2.memset(42.0f);
    pb2.copy!(Copy.deviceToHost);
    foreach (i; 0 .. W * H) enforce(pattern[i] == 42.0f);

    // Linear memset: 4-byte (cuMemsetD32) ...
    float[64] linf;
    auto b_f = Buffer!float(linf[]); scope(exit) b_f.release();
    b_f.memset(7.5f);
    b_f.copy!(Copy.deviceToHost);
    foreach (v; linf) enforce(v == 7.5f);

    // ... 2-byte (cuMemsetD16) ...
    ushort[64] lins;
    auto b_s = Buffer!ushort(lins[]); scope(exit) b_s.release();
    b_s.memset(0xBEEF);
    b_s.copy!(Copy.deviceToHost);
    foreach (v; lins) enforce(v == 0xBEEF);

    // ... and 1-byte (cuMemsetD8).
    ubyte[64] linb;
    auto b_b = Buffer!ubyte(linb[]); scope(exit) b_b.release();
    b_b.memset(0xAB);
    b_b.copy!(Copy.deviceToHost);
    foreach (v; linb) enforce(v == 0xAB);

    // 3D descriptor copy (copy3D/cuMemcpy3D): dense host cube ->
    // linear device buffer -> back into a second host cube.
    enum size_t W3 = 5, H3 = 3, D3 = 4;
    float[W3 * H3 * D3] cube, cubeBack;
    foreach (i; 0 .. cube.length) cube[i] = 1000 + i;
    auto b_3 = Buffer!float(cube.length); scope(exit) b_3.release();

    CUDA_MEMCPY3D up; // zero-initialised; unused fields must stay 0/null
    up.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
    up.srcHost       = cube.ptr;
    up.srcPitch      = W3 * float.sizeof;
    up.srcHeight     = H3;
    up.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
    up.dstDevice     = b_3.raw;
    up.dstPitch      = W3 * float.sizeof;
    up.dstHeight     = H3;
    up.WidthInBytes  = W3 * float.sizeof;
    up.Height        = H3;
    up.Depth         = D3;
    copy3D(up);

    CUDA_MEMCPY3D down;
    down.srcMemoryType = CUmemorytype.CU_MEMORYTYPE_DEVICE;
    down.srcDevice     = b_3.raw;
    down.srcPitch      = W3 * float.sizeof;
    down.srcHeight     = H3;
    down.dstMemoryType = CUmemorytype.CU_MEMORYTYPE_HOST;
    down.dstHost       = cubeBack.ptr;
    down.dstPitch      = W3 * float.sizeof;
    down.dstHeight     = H3;
    down.WidthInBytes  = W3 * float.sizeof;
    down.Height        = H3;
    down.Depth         = D3;
    copy3D(down);
    foreach (i; 0 .. cube.length) enforce(cubeBack[i] == 1000 + i);

    writeln("Pitched memory / 2D-3D copy / memset tests PASSED.");
}
