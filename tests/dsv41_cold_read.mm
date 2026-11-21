// Manual GPU-time control for the DSV41_COLD_READ step meter. The input covers one GiB of packed weights.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <fcntl.h>
#import <unistd.h>

typedef struct {
    int order, tg, loads;
} Arm;
static const Arm arms[] = {{0, 32, 16}, {1, 32, 16}, {1, 256, 16}, {2, 32, 16}, {3, 32, 16}, {4, 32, 16},
                           {5, 32, 16}, {1, 32, 4},  {1, 32, 32},  {1, 64, 8},  {1, 128, 8}};
static const char *body = R"msl(
#include <metal_stdlib>
using namespace metal;

kernel void read_weight(device const uint4 *a [[buffer(0)]], device const uint4 *b [[buffer(1)]],
                        device const uint4 *c [[buffer(2)]], device const uint4 *d [[buffer(3)]],
                        device uint *out [[buffer(4)]], uint t [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint n = VECTORS, threads = n / LOADS;
    uint4 acc = 0;
#pragma unroll
    for (uint k = 0; k < LOADS; k++) {
        uint block = t / 32;
#if ORDER == 0
        acc += a[t * LOADS + k];
#elif ORDER == 1
        acc += a[(block * LOADS + k) * 32 + lane];
#elif ORDER == 2
        acc += a[k * threads + t];
#elif ORDER == 3
        acc += a[(k % 2) * (n / 2) + (block * (LOADS / 2) + k / 2) * 32 + lane];
#else
        uint i = (block * (LOADS / 4) + k / 4) * 32 + lane;
#if ORDER == 4
        acc += a[(k % 4) * (n / 4) + i];
#else
        switch (k % 4) {
            case 0: acc += a[i]; break;
            case 1: acc += b[i]; break;
            case 2: acc += c[i]; break;
            default: acc += d[i]; break;
        }
#endif
#endif
    }
    uint sum = simd_sum(acc.x + acc.y + acc.z + acc.w);
    if (lane == 0) out[t / 32] = sum;
}
)msl";

int main(int argc, char **argv) {
    @autoreleasepool {
        if (!getenv("DSV41_COLD_READ") || (argc != 3 && argc != 4)) {
            fprintf(stderr, "Usage: DSV41_COLD_READ=1 %s packed_tensor_file byte_offset [MiB_per_dispatch]\n",
                    argv[0]);
            return 1;
        }
        const int mib = argc == 4 ? atoi(argv[3]) : 32;
        if (mib != 8 && mib != 16 && mib != 32 && mib != 64 && mib != 128)
            return 1;
        const int bytes = mib << 20;
        const int vectors = bytes / 16;
        const int bufferCount = 1024 / mib, dispatchCount = bufferCount * 4;
        int fd = open(argv[1], O_RDONLY);
        if (fd < 0)
            return 2;
        off_t offset = strtoll(argv[2], NULL, 10);
        id<MTLDevice> gpu = MTLCreateSystemDefaultDevice();
        if (!gpu)
            return 3;
        id<MTLCommandQueue> queue = [gpu newCommandQueue];
        id<MTLBuffer> full[128], split[128][4];
        uint32_t sums[128] = {};
        for (int i = 0; i < bufferCount; i++) {
            full[i] = [gpu newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (pread(fd, full[i].contents, bytes, offset + (off_t)i * bytes) != bytes)
                return 4;
            const uint32_t *p = (const uint32_t *)full[i].contents;
            for (int j = 0; j < bytes / 4; j++)
                sums[i] += p[j];
            for (int k = 0; k < 4; k++)
                split[i][k] = [gpu newBufferWithBytes:(char *)p + k * (bytes / 4)
                                               length:bytes / 4
                                              options:MTLResourceStorageModeShared];
        }
        close(fd);
        const int count = argc == 4 ? 1 : sizeof(arms) / sizeof(*arms);
        id<MTLComputePipelineState> pipelines[sizeof(arms) / sizeof(*arms)];
        for (int i = 0; i < count; i++) {
            NSError *error = nil;
            MTLCompileOptions *opts = [MTLCompileOptions new];
            opts.preprocessorMacros =
                @{@"ORDER" : @(arms[i].order),
                  @"VECTORS" : @(vectors),
                  @"LOADS" : @(arms[i].loads)};
            id<MTLLibrary> lib = [gpu newLibraryWithSource:[NSString stringWithUTF8String:body]
                                                   options:opts
                                                     error:&error];
            if (!lib) {
                NSLog(@"%@", error);
                return 5;
            }
            pipelines[i] = [gpu newComputePipelineStateWithFunction:[lib newFunctionWithName:@"read_weight"]
                                                              error:&error];
            if (!pipelines[i]) {
                NSLog(@"%@", error);
                return 6;
            }
        }
        const int stride = vectors / 4 / 32;
        id<MTLBuffer> out = [gpu newBufferWithLength:dispatchCount * stride * 4
                                             options:MTLResourceStorageModeShared];
        for (int pass = 0; pass < 6; pass++)
            for (int visit = 0; visit < 2 * count - 1; visit++)
                @autoreleasepool {
                    int armIndex = visit % 2 ? 1 + (visit / 2 + pass) % (count - 1) : 0;
                    Arm a = arms[armIndex];
                    unsigned threads = vectors / a.loads;
                    id<MTLCommandBuffer> cmd = [queue commandBuffer];
                    id<MTLComputeCommandEncoder> e =
                        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
                    [e setComputePipelineState:pipelines[armIndex]];
                    for (int j = 0; j < dispatchCount; j++) {
                        for (int k = 0; k < 4; k++)
                            [e setBuffer:(a.order == 5 ? split[j % bufferCount][k] : full[j % bufferCount])
                                  offset:0
                                 atIndex:k];
                        [e setBuffer:out offset:j * stride * 4 atIndex:4];
                        [e dispatchThreads:MTLSizeMake(threads, 1, 1)
                            threadsPerThreadgroup:MTLSizeMake(a.tg, 1, 1)];
                    }
                    [e endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    if (cmd.error) {
                        NSLog(@"%@", cmd.error);
                        return 7;
                    }
                    double us = (cmd.GPUEndTime - cmd.GPUStartTime) * 1e6 / dispatchCount;
                    for (int j = 0; j < dispatchCount; j++) {
                        uint32_t sum = 0;
                        uint32_t *p = (uint32_t *)out.contents + j * stride;
                        for (int k = 0; k < threads / 32; k++)
                            sum += p[k];
                        if (sum != sums[j % bufferCount]) {
                            fprintf(stderr, "Checksum mismatch, arm %d, dispatch %d\n", armIndex, j);
                            return 8;
                        }
                    }
                    printf("pass=%d arm=%d us=%.3f GB/s=%.2f\n", pass, armIndex, us, bytes / (1000 * us));
                    fflush(stdout);
                }
        return 0;
    }
}
