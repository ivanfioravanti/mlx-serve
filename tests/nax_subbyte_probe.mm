// nax_subbyte_probe.mm — probe MetalPerformancePrimitives matmul2d (NAX path)
// for sub-byte weight formats on Apple Silicon.
//
// Ladder: fp16 control, int8 control, uint4, int4, fp4_e2m1, int2.
// Per format a 4-stage verdict: MACRO (compiler feature macro present),
// COMPILE (runtime Metal source compile), PIPELINE (PSO creation on the
// default device), PARITY (match against a CPU fp32 reference).
//
// All format macro/type names arrive as -D defines from nax_subbyte_probe.sh,
// which greps them out of the SDK headers. Nothing here hardcodes a sub-byte
// spelling: PROBE_MACRO_<FMT> / PROBE_TYPE_<FMT> only.
//
// Build: see nax_subbyte_probe.sh.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#define PRB_STR2(x) #x
#define PRB_STR(x) PRB_STR2(x)
#define PRB_CAT2(a, b) a##b
#define PRB_CAT(a, b) PRB_CAT2(a, b)
#define PRB_MACRONAME(i) PRB_STR(PRB_CAT(PROBE_MACRO_, i))

#ifndef PROBE_MACRO_COUNT
#define PROBE_MACRO_COUNT 0
#endif

// ---------------------------------------------------------------------------
// Format table (names from -D defines; missing define => header-absent row)
// ---------------------------------------------------------------------------

enum FormatKind {
  K_HALF = 0, K_INT8 = 1, K_UINT4 = 2, K_INT4 = 3, K_FP4 = 4, K_INT2 = 5,
  K_FP8E4M3 = 6, K_FP8E5M2 = 7
};

struct FormatSpec {
  const char *key;
  const char *typeName;  // nullptr => absent from SDK headers
  const char *macroName; // nullptr => absent from SDK headers
  int kind;
  double tol;
};

static std::vector<FormatSpec> buildFormatTable() {
  std::vector<FormatSpec> v;
#if defined(PROBE_TYPE_FP16) && defined(PROBE_MACRO_FP16)
  v.push_back({"fp16 (control)", PRB_STR(PROBE_TYPE_FP16), PRB_STR(PROBE_MACRO_FP16), K_HALF, 1e-2});
#else
  v.push_back({"fp16 (control)", nullptr, nullptr, K_HALF, 1e-2});
#endif
#if defined(PROBE_TYPE_INT8) && defined(PROBE_MACRO_INT8)
  v.push_back({"int8 (control)", PRB_STR(PROBE_TYPE_INT8), PRB_STR(PROBE_MACRO_INT8), K_INT8, 1e-3});
#else
  v.push_back({"int8 (control)", nullptr, nullptr, K_INT8, 1e-3});
#endif
#if defined(PROBE_TYPE_UINT4) && defined(PROBE_MACRO_UINT4)
  v.push_back({"uint4", PRB_STR(PROBE_TYPE_UINT4), PRB_STR(PROBE_MACRO_UINT4), K_UINT4, 1e-3});
#else
  v.push_back({"uint4", nullptr, nullptr, K_UINT4, 1e-3});
#endif
#if defined(PROBE_TYPE_INT4) && defined(PROBE_MACRO_INT4)
  v.push_back({"int4", PRB_STR(PROBE_TYPE_INT4), PRB_STR(PROBE_MACRO_INT4), K_INT4, 1e-3});
#else
  v.push_back({"int4", nullptr, nullptr, K_INT4, 1e-3});
#endif
#if defined(PROBE_TYPE_FP4) && defined(PROBE_MACRO_FP4)
  v.push_back({"fp4_e2m1", PRB_STR(PROBE_TYPE_FP4), PRB_STR(PROBE_MACRO_FP4), K_FP4, 0.75});
#else
  v.push_back({"fp4_e2m1", nullptr, nullptr, K_FP4, 0.75});
#endif
#if defined(PROBE_TYPE_INT2) && defined(PROBE_MACRO_INT2)
  v.push_back({"int2", PRB_STR(PROBE_TYPE_INT2), PRB_STR(PROBE_MACRO_INT2), K_INT2, 1e-3});
#else
  v.push_back({"int2", nullptr, nullptr, K_INT2, 1e-3});
#endif
#if defined(PROBE_TYPE_FP8E4M3) && defined(PROBE_MACRO_FP8E4M3)
  v.push_back({"fp8_e4m3", PRB_STR(PROBE_TYPE_FP8E4M3), PRB_STR(PROBE_MACRO_FP8E4M3), K_FP8E4M3, 1.0});
#else
  v.push_back({"fp8_e4m3", nullptr, nullptr, K_FP8E4M3, 1.0});
#endif
#if defined(PROBE_TYPE_FP8E5M2) && defined(PROBE_MACRO_FP8E5M2)
  v.push_back({"fp8_e5m2", PRB_STR(PROBE_TYPE_FP8E5M2), PRB_STR(PROBE_MACRO_FP8E5M2), K_FP8E5M2, 1.0});
#else
  v.push_back({"fp8_e5m2", nullptr, nullptr, K_FP8E5M2, 1.0});
#endif
  return v;
}

// ---------------------------------------------------------------------------
// Language versions to try, newest first
// ---------------------------------------------------------------------------

struct LangVer {
  const char *label;
  NSUInteger value; // MTLLanguageVersion when !isDefault
  bool isDefault;
};

static std::vector<LangVer> buildLangVersions() {
  std::vector<LangVer> v;
  if (@available(macOS 27.0, *)) {
    v.push_back({"4.1", (NSUInteger)MTLLanguageVersion4_1, false});
  }
  if (@available(macOS 26.0, *)) {
    v.push_back({"4.0", (NSUInteger)MTLLanguageVersion4_0, false});
  }
  v.push_back({"default", 0, true});
  return v;
}

// ---------------------------------------------------------------------------
// Macro probing. One tiny compile per (macro, language version):
//   - #ifndef macro -> #error PRB_RESULT_UNDEFINED
//   - else an always-failing static_assert whose message is the stringized
//     macro value (static_assert messages are macro-expanded; #error is not).
// ---------------------------------------------------------------------------

struct MacroInfo {
  bool defined = false;
  bool available = false; // defined and value != "0"
  std::string value;      // stringized value, "" if defined-empty, "?" if unparsed
};

static NSString *firstDiagLine(NSError *err) {
  if (!err) return @"(no error info)";
  NSString *d = err.localizedDescription;
  NSArray<NSString *> *lines = [d componentsSeparatedByCharactersInSet:[NSCharacterSet newlineCharacterSet]];
  for (NSString *l in lines) {
    if (l.length) return l;
  }
  return @"(empty diagnostic)";
}

static MacroInfo probeMacro(id<MTLDevice> dev, const char *macro, const LangVer &lv) {
  MacroInfo info;
  NSString *src = [NSString stringWithFormat:
      @"#include <metal_stdlib>\n"
       "#include <metal_tensor>\n"
       "#define PRB_S2(x) #x\n"
       "#define PRB_S(x) PRB_S2(x)\n"
       "#ifndef %s\n"
       "#error PRB_RESULT_UNDEFINED\n"
       "#else\n"
       "static_assert(sizeof(char) != 1, PRB_S(%s));\n"
       "#endif\n"
       "kernel void prb_dummy_k() {}\n",
       macro, macro];
  MTLCompileOptions *opt = [MTLCompileOptions new];
  if (!lv.isDefault) opt.languageVersion = (MTLLanguageVersion)lv.value;
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
  NSString *diag = err ? err.localizedDescription : @"";
  if ([diag containsString:@"PRB_RESULT_UNDEFINED"]) {
    info.defined = false;
    info.value = "undef";
    return info;
  }
  info.defined = true;
  if (lib != nil) {
    // static_assert unexpectedly did not fire; macro is defined, value unknown.
    info.value = "?";
    info.available = true;
    return info;
  }
  NSRegularExpression *re =
      [NSRegularExpression regularExpressionWithPattern:@"\"([^\"]*)\"" options:0 error:nil];
  NSArray<NSTextCheckingResult *> *ms =
      [re matchesInString:diag options:0 range:NSMakeRange(0, diag.length)];
  if (ms.count) {
    info.value = [diag substringWithRange:[ms.lastObject rangeAtIndex:1]].UTF8String;
  } else {
    info.value = "?";
    fprintf(stderr, "note: could not parse value of %s (%s) from diagnostic:\n%s\n",
            macro, lv.label, diag.UTF8String);
  }
  info.available = (info.value != "0");
  return info;
}

// Cache: macro name -> per-language-version MacroInfo
static std::map<std::string, std::vector<MacroInfo>> g_macroCache;

static const MacroInfo &macroInfo(id<MTLDevice> dev, const char *macro, size_t li,
                                  const std::vector<LangVer> &langs) {
  auto it = g_macroCache.find(macro);
  if (it == g_macroCache.end()) {
    std::vector<MacroInfo> per(langs.size());
    for (size_t i = 0; i < langs.size(); ++i) per[i] = probeMacro(dev, macro, langs[i]);
    it = g_macroCache.emplace(macro, std::move(per)).first;
  }
  return it->second[li];
}

// ---------------------------------------------------------------------------
// Matmul kernel source for one format. Layout convention (matches the proven
// in-repo matmul2d usages): left A extents {K, M} strides {1, K} (row-major
// MxK halfs); right B extents {N, K} strides {1, N} (n contiguous); dest C
// extents {N, M} strides {1, N} (row-major MxN floats).
// ---------------------------------------------------------------------------

static NSString *kernelSource(const char *typeName) {
  return [NSString stringWithFormat:
      @"#include <metal_stdlib>\n"
       "#include <metal_tensor>\n"
       "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
       "using namespace metal;\n"
       "using namespace mpp::tensor_ops;\n"
       "kernel void nax_probe_kernel(\n"
       "    device const void *bufA [[buffer(0)]],\n"
       "    device const void *bufB [[buffer(1)]],\n"
       "    device float *bufC [[buffer(2)]],\n"
       "    constant int &Md [[buffer(3)]],\n"
       "    constant int &Nd [[buffer(4)]],\n"
       "    constant int &Kd [[buffer(5)]],\n"
       "    uint2 tgid [[threadgroup_position_in_grid]]) {\n"
       // Format-typed tensors use a byte-granular data handle (device uchar*);
       // take whatever handle type the tensor itself exposes.
       "  using TAt = tensor<device half, dextents<int, 2>, tensor_inline>;\n"
       "  using TBt = tensor<device %s, dextents<int, 2>, tensor_inline>;\n"
       "  using TCt = tensor<device float, dextents<int, 2>, tensor_inline>;\n"
       "  TAt tA((TAt::data_handle_type)bufA,\n"
       "      dextents<int, 2>(Kd, Md), array<int, 2>{1, Kd});\n"
       "  TBt tB((TBt::data_handle_type)bufB,\n"
       "      dextents<int, 2>(Nd, Kd), array<int, 2>{1, Nd});\n"
       "  TCt tC((TCt::data_handle_type)bufC,\n"
       "      dextents<int, 2>(Nd, Md), array<int, 2>{1, Nd});\n"
       "  auto sA = tA.slice(0, (int)tgid.y * 16);\n"
       "  auto sB = tB.slice((int)tgid.x * 16, 0);\n"
       "  auto sC = tC.slice((int)tgid.x * 16, (int)tgid.y * 16);\n"
       "  matmul2d<matmul2d_descriptor(16, 16, dynamic_extent), execution_simdgroups<4>> mm;\n"
       "  auto cT = mm.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();\n"
       "  mm.run(sA, sB, cT);\n"
       "  cT.store(sC);\n"
       "}\n",
       typeName];
}

// ---------------------------------------------------------------------------
// Host-side data generation, packing, CPU reference
// ---------------------------------------------------------------------------

static const int kM = 32, kN = 32, kK = 64; // K* N* = 2048 elems; 4b -> 1024 B, 2b -> 512 B

// e2m1 decode: 1 sign, 2 exp (bias 1), 1 mantissa. Subnormal e==0 -> 0.5*m.
static float dequantFP4(uint32_t raw) {
  int s = (raw >> 3) & 1, e = (raw >> 1) & 3, m = raw & 1;
  float mag = (e == 0) ? 0.5f * (float)m : (float)(1 << (e - 1)) * (1.0f + 0.5f * (float)m);
  return s ? -mag : mag;
}

// e4m3 decode: 1 sign, 4 exp (bias 7), 3 mantissa. IEEE-like subnormals
// (e==0 -> m * 2^-9). NO infinity: e==15 && m==7 is NaN. Max finite 448.
static float dequantFP8E4M3(uint32_t raw) {
  int s = (raw >> 7) & 1, e = (raw >> 3) & 0xF, m = raw & 7;
  if (e == 0xF && m == 7) return NAN;
  float mag = (e == 0) ? std::ldexpf((float)m, -9)
                      : std::ldexpf(1.0f + (float)m / 8.0f, e - 7);
  return s ? -mag : mag;
}

// e5m2 decode: 1 sign, 5 exp (bias 15), 2 mantissa. Subnormals e==0 ->
// m * 2^-16. e==31: m==0 is +-inf, m!=0 is NaN.
static float dequantFP8E5M2(uint32_t raw) {
  int s = (raw >> 7) & 1, e = (raw >> 2) & 0x1F, m = raw & 3;
  if (e == 0x1F) return (m == 0) ? INFINITY : NAN;
  float mag = (e == 0) ? std::ldexpf((float)m, -16)
                      : std::ldexpf(1.0f + (float)m / 4.0f, e - 15);
  return s ? -mag : mag;
}

// Curated fp8 test encodings: all codes whose decoded value is finite (NaN
// and inf encodings excluded) and a multiple of 0.25 within [-8, +8]. Those
// quarters are exactly representable in BOTH e4m3 and e5m2, and A(int) *
// B(quarter) sums stay exact in fp32 at both probe and bench shapes.
// Subnormal codes exist in both formats (decode handles them) but are left
// out of the test set on purpose: mixing 2^-9/2^-16 values with +-8 sums
// would break fp32 exactness of the CPU reference.
static const std::vector<uint8_t> &fp8Codes(int kind) {
  static std::vector<uint8_t> e4m3Codes, e5m2Codes;
  std::vector<uint8_t> *v = (kind == K_FP8E4M3) ? &e4m3Codes : &e5m2Codes;
  if (v->empty()) {
    for (int raw = 0; raw < 256; ++raw) {
      float val = (kind == K_FP8E4M3) ? dequantFP8E4M3((uint32_t)raw)
                                      : dequantFP8E5M2((uint32_t)raw);
      if (!std::isfinite(val)) continue;   // NaN / inf encodings
      if (std::fabs(val) > 8.0f) continue; // keep reference sums fp32-exact
      float q = val * 4.0f;
      if (q != std::floor(q)) continue;    // quarters only (exact products)
      v->push_back((uint8_t)raw);
    }
  }
  return *v;
}

static float dequantRaw(int kind, uint32_t raw) {
  switch (kind) {
    case K_UINT4: return (float)(raw & 0xF);
    case K_INT4: { int x = raw & 0xF; return (float)(x >= 8 ? x - 16 : x); }
    case K_INT2: { int x = raw & 0x3; return (float)(x >= 2 ? x - 4 : x); }
    case K_FP4:  return dequantFP4(raw & 0xF);
    case K_FP8E4M3: return dequantFP8E4M3(raw & 0xFF);
    case K_FP8E5M2: return dequantFP8E5M2(raw & 0xFF);
    default: return 0.0f;
  }
}

// Deterministic full-range raw code for element index idx.
static uint32_t rawCode(int kind, int idx) {
  switch (kind) {
    case K_UINT4: case K_INT4: case K_FP4: return (uint32_t)(idx % 16);
    case K_INT2: return (uint32_t)(idx % 4);
    default: return 0;
  }
}

// Pack `count` raw codes of `bits` each. order 0: first element in least
// significant field; order 1: first element in most significant field.
static void packSubByte(uint8_t *out, int count, int bits, int order,
                        const std::vector<uint32_t> &raws) {
  int per = 8 / bits;
  memset(out, 0, (size_t)(count + per - 1) / per);
  for (int i = 0; i < count; ++i) {
    int slot = i % per;
    int shift = (order == 0) ? slot * bits : (per - 1 - slot) * bits;
    out[i / per] |= (uint8_t)((raws[i] & ((1u << bits) - 1u)) << shift);
  }
}

static bool runKernel(id<MTLDevice> dev, id<MTLComputePipelineState> pso,
                      const void *Abytes, size_t Alen, const void *Bbytes, size_t Blen,
                      float *Cout, std::string &err) {
  @autoreleasepool {
    id<MTLBuffer> bufA = [dev newBufferWithBytes:Abytes length:Alen options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufB = [dev newBufferWithBytes:Bbytes length:Blen options:MTLResourceStorageModeShared];
    id<MTLBuffer> bufC = [dev newBufferWithLength:(NSUInteger)kM * kN * sizeof(float)
                                          options:MTLResourceStorageModeShared];
    if (!bufA || !bufB || !bufC) { err = "buffer allocation failed"; return false; }
    memset(bufC.contents, 0xEE, (size_t)kM * kN * sizeof(float)); // poison
    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    int Md = kM, Nd = kN, Kd = kK;
    [enc setBytes:&Md length:sizeof(int) atIndex:3];
    [enc setBytes:&Nd length:sizeof(int) atIndex:4];
    [enc setBytes:&Kd length:sizeof(int) atIndex:5];
    NSUInteger w = pso.threadExecutionWidth;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)(kN / 16), (NSUInteger)(kM / 16), 1)
        threadsPerThreadgroup:MTLSizeMake(w * 4, 1, 1)];
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { err = cb.error.localizedDescription.UTF8String; return false; }
    memcpy(Cout, bufC.contents, (size_t)kM * kN * sizeof(float));
    return true;
  }
}

// Build A (half activations), B bytes for the given packing order, and the
// fp32 reference. Returns max abs error, or <0 on harness error.
static double parityCheck(id<MTLDevice> dev, id<MTLComputePipelineState> pso, const FormatSpec &f,
                          int order, std::string &err) {
  // Activations: exact-in-half small ints in [-4, 4].
  std::vector<_Float16> Ah((size_t)kM * kK);
  std::vector<float> Aval((size_t)kM * kK);
  for (int m = 0; m < kM; ++m)
    for (int k = 0; k < kK; ++k) {
      Ah[(size_t)m * kK + k] = (_Float16)((m * 7 + k * 13) % 9 - 4);
      Aval[(size_t)m * kK + k] = (float)Ah[(size_t)m * kK + k];
    }

  std::vector<uint8_t> Bbytes;
  std::vector<float> Bval((size_t)kK * kN);
  const int elems = kK * kN;
  switch (f.kind) {
    case K_HALF: {
      std::vector<_Float16> Bh((size_t)elems);
      for (int i = 0; i < elems; ++i) {
        Bh[i] = (_Float16)((i * 5) % 7 - 3);
        Bval[i] = (float)Bh[i];
      }
      Bbytes.resize((size_t)elems * 2);
      memcpy(Bbytes.data(), Bh.data(), Bbytes.size());
      break;
    }
    case K_INT8: {
      std::vector<int8_t> B8((size_t)elems);
      for (int i = 0; i < elems; ++i) {
        B8[i] = (int8_t)(i % 7 - 3);
        Bval[i] = (float)B8[i];
      }
      Bbytes.resize((size_t)elems);
      memcpy(Bbytes.data(), B8.data(), Bbytes.size());
      break;
    }
    case K_UINT4: case K_INT4: case K_FP4: case K_INT2: {
      int bits = (f.kind == K_INT2) ? 2 : 4;
      std::vector<uint32_t> raws((size_t)elems);
      for (int i = 0; i < elems; ++i) {
        raws[i] = rawCode(f.kind, i);
        Bval[i] = dequantRaw(f.kind, raws[i]);
      }
      Bbytes.resize((size_t)elems * bits / 8);
      packSubByte(Bbytes.data(), elems, bits, order, raws);
      break;
    }
    case K_FP8E4M3: case K_FP8E5M2: {
      const std::vector<uint8_t> &codes = fp8Codes(f.kind);
      Bbytes.resize((size_t)elems);
      for (int i = 0; i < elems; ++i) {
        uint8_t raw = codes[(size_t)i % codes.size()];
        Bbytes[(size_t)i] = raw;
        Bval[i] = dequantRaw(f.kind, raw);
      }
      break;
    }
  }

  std::vector<float> Cgpu((size_t)kM * kN);
  if (!runKernel(dev, pso, Ah.data(), Ah.size() * 2, Bbytes.data(), Bbytes.size(),
                 Cgpu.data(), err))
    return -1.0;

  double maxErr = 0.0;
  for (int m = 0; m < kM; ++m)
    for (int n = 0; n < kN; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < kK; ++k)
        acc += Aval[(size_t)m * kK + k] * Bval[(size_t)k * kN + n];
      double d = std::fabs((double)Cgpu[(size_t)m * kN + n] - (double)acc);
      if (d > maxErr) maxErr = d;
    }
  return maxErr;
}

// ---------------------------------------------------------------------------
// Per-format stage results
// ---------------------------------------------------------------------------

struct Verdict {
  std::string macroS = "—", compileS = "—", pipelineS = "—", parityS = "—";
  bool macroOK = false, compileOK = false, pipelineOK = false, parityOK = false;
};

static const char *packOrderName(int kind, int order) {
  if (kind == K_INT2) return order == 0 ? "lsb-first" : "msb-first";
  return order == 0 ? "lo-nibble-first" : "hi-nibble-first";
}

// ===========================================================================
// --bench mode: throughput ladder (hardware-vs-software experiment).
// Same shapes for every format: M=N=K=4096, A half, B in-format, C float.
// matmul2d runs in mode::multiply so each iteration fully overwrites C
// (no re-zeroing between iterations). Timing via MTLCommandBuffer GPU
// timestamps with a wall-clock fallback. Every bench is guarded by a CPU
// checksum + a 16x16 corner reference so a kernel that does not write C is
// caught instead of reported as fast.
// ===========================================================================

static const int kBM = 4096, kBN = 4096, kBK = 4096;
static const int kBenchWarmup = 3, kBenchIters = 10;   // matmul2d
static const int kSWWarmup = 2, kSWIters = 3;          // software baselines

// Fast inside-tile matmul2d kernel: static_slice, no edge checking. Grid must
// divide evenly (4096 % 64 == 0, 4096 % 32 == 0, 4096 % 16 == 0).
static NSString *benchKernelSource(const char *typeName, int TM, int TN) {
  return [NSString stringWithFormat:
      @"#include <metal_stdlib>\n"
       "#include <metal_tensor>\n"
       "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>\n"
       "using namespace metal;\n"
       "using namespace mpp::tensor_ops;\n"
       "kernel void nax_bench_kernel(\n"
       "    device const void *bufA [[buffer(0)]],\n"
       "    device const void *bufB [[buffer(1)]],\n"
       "    device float *bufC [[buffer(2)]],\n"
       "    constant int &Md [[buffer(3)]],\n"
       "    constant int &Nd [[buffer(4)]],\n"
       "    constant int &Kd [[buffer(5)]],\n"
       "    uint2 tgid [[threadgroup_position_in_grid]]) {\n"
       "  using TAt = tensor<device half, dextents<int, 2>, tensor_inline>;\n"
       "  using TBt = tensor<device %s, dextents<int, 2>, tensor_inline>;\n"
       "  using TCt = tensor<device float, dextents<int, 2>, tensor_inline>;\n"
       "  TAt tA((TAt::data_handle_type)bufA,\n"
       "      dextents<int, 2>(Kd, Md), array<int, 2>{1, Kd});\n"
       "  TBt tB((TBt::data_handle_type)bufB,\n"
       "      dextents<int, 2>(Nd, Kd), array<int, 2>{1, Nd});\n"
       "  TCt tC((TCt::data_handle_type)bufC,\n"
       "      dextents<int, 2>(Nd, Md), array<int, 2>{1, Nd});\n"
       "  auto sA = tA.slice<dynamic_extent, %d>(0, (int)tgid.y * %d);\n"
       "  auto sB = tB.slice<%d, dynamic_extent>((int)tgid.x * %d, 0);\n"
       "  auto sC = tC.slice<%d, %d>((int)tgid.x * %d, (int)tgid.y * %d);\n"
       "  matmul2d<matmul2d_descriptor(%d, %d, dynamic_extent), execution_simdgroups<4>> mm;\n"
       "  auto cT = mm.get_destination_cooperative_tensor<decltype(sA), decltype(sB), float>();\n"
       "  mm.run(sA, sB, cT);\n"
       "  cT.store(sC);\n"
       "}\n",
       typeName, TM, TM, TN, TN, TN, TM, TN, TM, TM, TN];
}

// Software baselines: plain compute shaders, identical math, no tensor ops.
// Each thread produces 4 consecutive N elements of one M row.
static const char *kSWCommonHeader =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n";

static NSString *kSWFP16Source = [NSString stringWithFormat:
    @"%s"
     "kernel void sw_fp16_fma(\n"
     "    device const half *A [[buffer(0)]],\n"
     "    device const void *Bv [[buffer(1)]],\n"
     "    device float *C [[buffer(2)]],\n"
     "    constant int &Md [[buffer(3)]],\n"
     "    constant int &Nd [[buffer(4)]],\n"
     "    constant int &Kd [[buffer(5)]],\n"
     "    uint2 tid [[thread_position_in_grid]]) {\n"
     "  (void)Md;\n"
     "  device const half *B = (device const half *)Bv;\n"
     "  const int m = (int)tid.y;\n"
     "  const int n0 = (int)tid.x * 4;\n"
     "  device const half *Arow = A + (long)m * Kd;\n"
     "  float4 acc = float4(0.0f);\n"
     "  for (int k = 0; k < Kd; ++k) {\n"
     "    float a = float(Arow[k]);\n"
     "    half4 bh = *(device const half4 *)(B + (long)k * Nd + n0);\n"
     "    acc += a * float4(bh);\n"
     "  }\n"
     "  device float *cp = C + (long)m * Nd + n0;\n"
     "  cp[0] = acc.x; cp[1] = acc.y; cp[2] = acc.z; cp[3] = acc.w;\n"
     "}\n",
     kSWCommonHeader];

static NSString *kSWUint4Source = [NSString stringWithFormat:
    @"%s"
     "kernel void sw_uint4_unpack_fma(\n"
     "    device const half *A [[buffer(0)]],\n"
     "    device const uchar *B [[buffer(1)]],\n"
     "    device float *C [[buffer(2)]],\n"
     "    constant int &Md [[buffer(3)]],\n"
     "    constant int &Nd [[buffer(4)]],\n"
     "    constant int &Kd [[buffer(5)]],\n"
     "    uint2 tid [[thread_position_in_grid]]) {\n"
     "  (void)Md;\n"
     "  const int m = (int)tid.y;\n"
     "  const int n0 = (int)tid.x * 4;\n"
     "  const int browBytes = Nd / 2;\n"
     "  device const half *Arow = A + (long)m * Kd;\n"
     "  float4 acc = float4(0.0f);\n"
     "  for (int k = 0; k < Kd; ++k) {\n"
     "    float a = float(Arow[k]);\n"
     "    uchar2 pk = *(device const uchar2 *)(B + (long)k * browBytes + n0 / 2);\n"
     "    float4 b = float4((float)(pk.x & 0xF), (float)(pk.x >> 4),\n"
     "                      (float)(pk.y & 0xF), (float)(pk.y >> 4));\n"
     "    acc += a * b;\n"
     "  }\n"
     "  device float *cp = C + (long)m * Nd + n0;\n"
     "  cp[0] = acc.x; cp[1] = acc.y; cp[2] = acc.z; cp[3] = acc.w;\n"
     "}\n",
     kSWCommonHeader];

static NSString *kSWInt2Source = [NSString stringWithFormat:
    @"%s"
     "kernel void sw_int2_unpack_fma(\n"
     "    device const half *A [[buffer(0)]],\n"
     "    device const uchar *B [[buffer(1)]],\n"
     "    device float *C [[buffer(2)]],\n"
     "    constant int &Md [[buffer(3)]],\n"
     "    constant int &Nd [[buffer(4)]],\n"
     "    constant int &Kd [[buffer(5)]],\n"
     "    uint2 tid [[thread_position_in_grid]]) {\n"
     "  (void)Md;\n"
     "  const int m = (int)tid.y;\n"
     "  const int n0 = (int)tid.x * 4;\n"
     "  const int browBytes = Nd / 4;\n"
     "  device const half *Arow = A + (long)m * Kd;\n"
     "  float4 acc = float4(0.0f);\n"
     "  for (int k = 0; k < Kd; ++k) {\n"
     "    float a = float(Arow[k]);\n"
     "    uchar pk = B[(long)k * browBytes + n0 / 4];\n"
     "    int x0 = pk & 3, x1 = (pk >> 2) & 3, x2 = (pk >> 4) & 3, x3 = (pk >> 6) & 3;\n"
     "    float4 b = float4((float)(x0 >= 2 ? x0 - 4 : x0), (float)(x1 >= 2 ? x1 - 4 : x1),\n"
     "                      (float)(x2 >= 2 ? x2 - 4 : x2), (float)(x3 >= 2 ? x3 - 4 : x3));\n"
     "    acc += a * b;\n"
     "  }\n"
     "  device float *cp = C + (long)m * Nd + n0;\n"
     "  cp[0] = acc.x; cp[1] = acc.y; cp[2] = acc.z; cp[3] = acc.w;\n"
     "}\n",
     kSWCommonHeader];

struct BenchResult {
  bool ok = false;          // ran and passed the checksum guard
  bool gpuTime = true;      // false => wall-clock fallback was used
  double bestMs = 0, medianMs = 0, tflops = 0;
  double checksum = 0, cornerErr = -1;
  int tileM = 0, tileN = 0;
  int warmup = 0, iters = 0;
  std::string note;
};

// Deterministic B value at flat index idx = k*N + n (same patterns as parity).
static float benchBval(int kind, long idx) {
  switch (kind) {
    case K_HALF: return (float)((idx * 5) % 7 - 3);
    case K_INT8: return (float)(idx % 7 - 3);
    case K_UINT4: case K_INT4: case K_FP4: return dequantRaw(kind, (uint32_t)(idx % 16));
    case K_INT2: return dequantRaw(kind, (uint32_t)(idx % 4));
    case K_FP8E4M3: case K_FP8E5M2: {
      const std::vector<uint8_t> &codes = fp8Codes(kind);
      return dequantRaw(kind, codes[(size_t)idx % codes.size()]);
    }
    default: return 0.0f;
  }
}

static id<MTLComputePipelineState> compilePSO(id<MTLDevice> dev, NSString *src,
                                              const char *fnName, const LangVer &lv,
                                              std::string &errOut) {
  MTLCompileOptions *opt = [MTLCompileOptions new];
  if (!lv.isDefault) opt.languageVersion = (MTLLanguageVersion)lv.value;
  NSError *err = nil;
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opt error:&err];
  if (!lib) {
    errOut = std::string("compile: ") + firstDiagLine(err).UTF8String;
    return nil;
  }
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:fnName]];
  if (!fn) { errOut = "function missing"; return nil; }
  err = nil;
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!pso) {
    errOut = std::string("pipeline: ") + (err ? firstDiagLine(err).UTF8String : "(unknown)");
    return nil;
  }
  return pso;
}

// One dispatch in its own command buffer; returns ms, or -1 on error.
static double runOneIter(id<MTLCommandQueue> q, id<MTLComputePipelineState> pso,
                         id<MTLBuffer> bufA, id<MTLBuffer> bufB, id<MTLBuffer> bufC,
                         MTLSize grid, MTLSize tpg, BOOL dispatchThreads,
                         BOOL *usedGPU, std::string &err) {
  @autoreleasepool {
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    [enc setComputePipelineState:pso];
    [enc setBuffer:bufA offset:0 atIndex:0];
    [enc setBuffer:bufB offset:0 atIndex:1];
    [enc setBuffer:bufC offset:0 atIndex:2];
    int Md = kBM, Nd = kBN, Kd = kBK;
    [enc setBytes:&Md length:sizeof(int) atIndex:3];
    [enc setBytes:&Nd length:sizeof(int) atIndex:4];
    [enc setBytes:&Kd length:sizeof(int) atIndex:5];
    if (dispatchThreads)
      [enc dispatchThreads:grid threadsPerThreadgroup:tpg];
    else
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tpg];
    [enc endEncoding];
    CFAbsoluteTime wall0 = CFAbsoluteTimeGetCurrent();
    [cb commit];
    [cb waitUntilCompleted];
    CFAbsoluteTime wall1 = CFAbsoluteTimeGetCurrent();
    if (cb.error) { err = cb.error.localizedDescription.UTF8String; return -1.0; }
    double t0 = cb.GPUStartTime, t1 = cb.GPUEndTime;
    if (t0 > 0 && t1 > t0) { *usedGPU = YES; return (t1 - t0) * 1000.0; }
    *usedGPU = NO;
    return (wall1 - wall0) * 1000.0;
  }
}

// Checksum + corner reference against the deterministic input patterns.
static void checksumAndCorner(const float *C, int kind, double *sumOut, double *cornerErrOut) {
  double sum = 0.0;
  const long total = (long)kBM * kBN;
  for (long i = 0; i < total; ++i) sum += (double)C[i];
  *sumOut = sum;
  double maxErr = 0.0;
  for (int m = 0; m < 16; ++m)
    for (int n = 0; n < 16; ++n) {
      float acc = 0.0f;
      for (int k = 0; k < kBK; ++k)
        acc += (float)((m * 7 + k * 13) % 9 - 4) * benchBval(kind, (long)k * kBN + n);
      double d = std::fabs((double)C[(long)m * kBN + n] - (double)acc);
      if (d > maxErr) maxErr = d;
    }
  *cornerErrOut = maxErr;
}

static void fillBenchA(std::vector<_Float16> &Ah) {
  Ah.resize((size_t)kBM * kBK);
  for (int m = 0; m < kBM; ++m)
    for (int k = 0; k < kBK; ++k)
      Ah[(size_t)m * kBK + k] = (_Float16)((m * 7 + k * 13) % 9 - 4);
}

static std::vector<uint8_t> fillBenchBBytes(int kind) {
  const long elems = (long)kBK * kBN;
  std::vector<uint8_t> bytes;
  switch (kind) {
    case K_HALF: {
      std::vector<_Float16> Bh((size_t)elems);
      for (long i = 0; i < elems; ++i) Bh[i] = (_Float16)((i * 5) % 7 - 3);
      bytes.resize((size_t)elems * 2);
      memcpy(bytes.data(), Bh.data(), bytes.size());
      break;
    }
    case K_INT8: {
      bytes.resize((size_t)elems);
      for (long i = 0; i < elems; ++i) bytes[i] = (uint8_t)(int8_t)(i % 7 - 3);
      break;
    }
    case K_UINT4: case K_INT4: case K_FP4: case K_INT2: {
      int bits = (kind == K_INT2) ? 2 : 4;
      std::vector<uint32_t> raws((size_t)elems);
      for (long i = 0; i < elems; ++i) raws[i] = rawCode(kind, (int)(i % 16));
      bytes.resize((size_t)elems * bits / 8);
      packSubByte(bytes.data(), (int)elems, bits, 0 /* lo-first, proven */, raws);
      break;
    }
    case K_FP8E4M3: case K_FP8E5M2: {
      const std::vector<uint8_t> &codes = fp8Codes(kind);
      bytes.resize((size_t)elems);
      for (long i = 0; i < elems; ++i) bytes[(size_t)i] = codes[(size_t)i % codes.size()];
      break;
    }
  }
  return bytes;
}

static BenchResult benchMatmul2D(id<MTLDevice> dev, id<MTLCommandQueue> q, const FormatSpec &f,
                                 const LangVer &lv, id<MTLBuffer> bufA,
                                 const std::vector<uint8_t> &Bbytes, id<MTLBuffer> bufC) {
  BenchResult r;
  r.warmup = kBenchWarmup; r.iters = kBenchIters;
  static const int kTiles[3][2] = {{64, 32}, {32, 32}, {16, 16}};
  id<MTLComputePipelineState> pso = nil;
  std::string err, lastErr;
  for (auto &t : kTiles) {
    pso = compilePSO(dev, benchKernelSource(f.typeName, t[0], t[1]), "nax_bench_kernel", lv, err);
    if (pso) { r.tileM = t[0]; r.tileN = t[1]; break; }
    lastErr = err;
  }
  if (!pso) { r.note = "bench kernel failed at 64x32/32x32/16x16: " + lastErr; return r; }

  id<MTLBuffer> bufB = [dev newBufferWithBytes:Bbytes.data() length:Bbytes.size()
                                       options:MTLResourceStorageModeShared];
  if (!bufB) { r.note = "B buffer alloc failed"; return r; }
  memset(bufC.contents, 0xEE, (size_t)kBM * kBN * sizeof(float)); // poison

  MTLSize grid = MTLSizeMake((NSUInteger)(kBN / r.tileN), (NSUInteger)(kBM / r.tileM), 1);
  MTLSize tpg = MTLSizeMake(pso.threadExecutionWidth * 4, 1, 1);

  std::vector<double> times;
  BOOL gpu = YES;
  for (int i = 0; i < r.warmup + r.iters; ++i) {
    double ms = runOneIter(q, pso, bufA, bufB, bufC, grid, tpg, NO, &gpu, err);
    if (ms < 0) { r.note = "dispatch: " + err; return r; }
    if (!gpu) r.gpuTime = false;
    if (i >= r.warmup) times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  r.bestMs = times.front();
  r.medianMs = times[times.size() / 2];
  r.tflops = 2.0 * (double)kBM * kBN * kBK / (r.medianMs * 1e-3) / 1e12;

  checksumAndCorner((const float *)bufC.contents, f.kind, &r.checksum, &r.cornerErr);
  if (r.cornerErr > f.tol) {
    r.note = "checksum guard failed (corner err exceeds tol)";
    return r;
  }
  r.ok = true;
  return r;
}

static BenchResult benchSW(id<MTLDevice> dev, id<MTLCommandQueue> q, NSString *src,
                           const char *fnName, int kind, const LangVer &lv, id<MTLBuffer> bufA,
                           const std::vector<uint8_t> &Bbytes, id<MTLBuffer> bufC, double tol) {
  BenchResult r;
  r.warmup = kSWWarmup; r.iters = kSWIters;
  std::string err;
  id<MTLComputePipelineState> pso = compilePSO(dev, src, fnName, lv, err);
  if (!pso) { r.note = err; return r; }
  id<MTLBuffer> bufB = [dev newBufferWithBytes:Bbytes.data() length:Bbytes.size()
                                       options:MTLResourceStorageModeShared];
  if (!bufB) { r.note = "B buffer alloc failed"; return r; }
  memset(bufC.contents, 0xEE, (size_t)kBM * kBN * sizeof(float));

  MTLSize grid = MTLSizeMake((NSUInteger)(kBN / 4), (NSUInteger)kBM, 1);
  MTLSize tpg = MTLSizeMake(32, 4, 1);

  std::vector<double> times;
  BOOL gpu = YES;
  for (int i = 0; i < r.warmup + r.iters; ++i) {
    double ms = runOneIter(q, pso, bufA, bufB, bufC, grid, tpg, YES, &gpu, err);
    if (ms < 0) { r.note = "dispatch: " + err; return r; }
    if (!gpu) r.gpuTime = false;
    if (i >= r.warmup) times.push_back(ms);
  }
  std::sort(times.begin(), times.end());
  r.bestMs = times.front();
  r.medianMs = times[times.size() / 2];
  r.tflops = 2.0 * (double)kBM * kBN * kBK / (r.medianMs * 1e-3) / 1e12;

  checksumAndCorner((const float *)bufC.contents, kind, &r.checksum, &r.cornerErr);
  if (r.cornerErr > tol) {
    r.note = "checksum guard failed (corner err exceeds tol)";
    return r;
  }
  r.ok = true;
  return r;
}

static void printBenchEvidence(const char *name, const BenchResult &r, double tol) {
  printf("== %s ==\n", name);
  if (!r.ok) {
    printf("  — %s\n\n", r.note.empty() ? "unavailable" : r.note.c_str());
    return;
  }
  printf("  warmup %d, timed %d iters (%s)\n", r.warmup, r.iters,
         r.gpuTime ? "GPU timestamps" : "WALL time (GPU timestamps unavailable)");
  if (r.tileM) printf("  tile %dx%d, multiply mode (C overwritten each iter)\n", r.tileM, r.tileN);
  printf("  best %.3f ms   median %.3f ms   -> %.3f TFLOP/s (median)\n", r.bestMs, r.medianMs,
         r.tflops);
  printf("  checksum %.9e   corner(16x16) max err %.6g (tol %.3g)   GUARD: PASS\n\n", r.checksum,
         r.cornerErr, tol);
}

static int runBench(id<MTLDevice> device) {
  printf("== Bench mode ==\n");
  printf("device: %s\n", device.name.UTF8String);
  printf("shape: M=N=K=%d; A half, B in-format (packed sub-byte), C float\n", kBM);
  printf("matmul2d: mode::multiply, C overwritten each iteration (no re-zeroing)\n\n");

  std::vector<LangVer> langs = buildLangVersions();
  std::vector<FormatSpec> formats = buildFormatTable();
  id<MTLCommandQueue> q = [device newCommandQueue];
  if (!q) {
    fprintf(stderr, "HARNESS FAILURE: no command queue\n");
    return 2;
  }

  // Shared A (identical for every format) and one reused C buffer.
  std::vector<_Float16> Ah;
  fillBenchA(Ah);
  id<MTLBuffer> bufA = [device newBufferWithBytes:Ah.data() length:Ah.size() * 2
                                          options:MTLResourceStorageModeShared];
  id<MTLBuffer> bufC = [device newBufferWithLength:(NSUInteger)kBM * kBN * sizeof(float)
                                           options:MTLResourceStorageModeShared];
  if (!bufA || !bufC) {
    fprintf(stderr, "HARNESS FAILURE: buffer allocation\n");
    return 2;
  }

  // B bytes per kind, built once and shared between matmul2d and SW benches.
  std::map<int, std::vector<uint8_t>> bBytes;
  for (auto &f : formats)
    if (f.typeName && !bBytes.count(f.kind)) bBytes[f.kind] = fillBenchBBytes(f.kind);

  // --- matmul2d benches ---
  std::vector<BenchResult> mmr(formats.size());
  std::vector<std::string> mmLang(formats.size());
  for (size_t fi = 0; fi < formats.size(); ++fi) {
    const FormatSpec &f = formats[fi];
    if (!f.typeName || !f.macroName) {
      mmr[fi].note = "header-absent";
      printBenchEvidence((std::string("matmul2d bench: ") + f.key).c_str(), mmr[fi], f.tol);
      continue;
    }
    int chosen = -1;
    for (size_t li = 0; li < langs.size(); ++li)
      if (macroInfo(device, f.macroName, li, langs).available) { chosen = (int)li; break; }
    if (chosen < 0) {
      mmr[fi].note = std::string("macro ") + f.macroName + " unavailable in all language versions";
      printBenchEvidence((std::string("matmul2d bench: ") + f.key).c_str(), mmr[fi], f.tol);
      continue;
    }
    mmLang[fi] = langs[chosen].label;
    mmr[fi] = benchMatmul2D(device, q, f, langs[chosen], bufA, bBytes[f.kind], bufC);
    printBenchEvidence((std::string("matmul2d bench: ") + f.key + " (lang " + mmLang[fi] + ")").c_str(),
                       mmr[fi], f.tol);
  }

  // --- software baselines ---
  const LangVer &defLang = langs.back(); // default language version is plenty for plain Metal
  auto hasB = [&](int kind) { return bBytes.count(kind) && !bBytes[kind].empty(); };
  BenchResult swFP16, swU4, swI2;
  if (hasB(K_HALF))
    swFP16 = benchSW(device, q, kSWFP16Source, "sw_fp16_fma", K_HALF, defLang, bufA,
                     bBytes[K_HALF], bufC, 1e-2);
  else
    swFP16.note = "fp16 B data unavailable (format header-absent)";
  printBenchEvidence("SW baseline: fp16_fma", swFP16, 1e-2);
  if (hasB(K_UINT4))
    swU4 = benchSW(device, q, kSWUint4Source, "sw_uint4_unpack_fma", K_UINT4, defLang,
                   bufA, bBytes[K_UINT4], bufC, 1e-3);
  else
    swU4.note = "uint4 B data unavailable (format header-absent)";
  printBenchEvidence("SW baseline: uint4_unpack_fma", swU4, 1e-3);
  if (hasB(K_INT2))
    swI2 = benchSW(device, q, kSWInt2Source, "sw_int2_unpack_fma", K_INT2, defLang, bufA,
                   bBytes[K_INT2], bufC, 1e-3);
  else
    swI2.note = "int2 B data unavailable (format header-absent)";
  printBenchEvidence("SW baseline: int2_unpack_fma", swI2, 1e-3);

  // SW baseline mapping per format kind (proxy baselines are printed as such).
  struct SWMap { const BenchResult *r; const char *name; };
  auto swFor = [&](int kind) -> SWMap {
    switch (kind) {
      case K_HALF: return {&swFP16, "fp16_fma"};
      case K_INT8: return {&swFP16, "fp16_fma (proxy)"};
      case K_UINT4: return {&swU4, "uint4_unpack_fma"};
      case K_INT4: return {&swU4, "uint4_unpack_fma (proxy)"};
      case K_FP4: return {&swU4, "uint4_unpack_fma (proxy)"};
      case K_INT2: return {&swI2, "int2_unpack_fma"};
      case K_FP8E4M3: return {&swFP16, "fp16_fma (proxy)"};
      case K_FP8E5M2: return {&swFP16, "fp16_fma (proxy)"};
      default: return {nullptr, "—"};
    }
  };

  // --- ladder table ---
  bool anyWall = !swFP16.gpuTime || !swU4.gpuTime || !swI2.gpuTime;
  for (auto &r : mmr) if (r.ok && !r.gpuTime) anyWall = true;
  printf("== Throughput ladder (M=N=K=%d, multiply mode, %s) ==\n", kBM,
         anyWall ? "mixed GPU/wall timestamps (see rows)" : "GPU timestamps");
  printf("%-12s %-17s %-9s %-8s %-26s %-13s %s\n", "FORMAT", "matmul2d TFLOP/s", "vs fp16", "tile",
         "SW baseline", "SW TFLOP/s", "matmul2d / SW");
  double fp16T = mmr[0].ok ? mmr[0].tflops : 0;
  for (size_t fi = 0; fi < formats.size(); ++fi) {
    const FormatSpec &f = formats[fi];
    const BenchResult &r = mmr[fi];
    SWMap sw = swFor(f.kind);
    char tfl[32], vs[32], tile[16], swt[32], ratio[32];
    if (r.ok) {
      snprintf(tfl, sizeof(tfl), "%.3f", r.tflops);
      if (fp16T > 0) snprintf(vs, sizeof(vs), "%.2fx", r.tflops / fp16T);
      else snprintf(vs, sizeof(vs), "—");
      snprintf(tile, sizeof(tile), "%dx%d", r.tileM, r.tileN);
    } else if (r.cornerErr >= 0 && r.note.find("guard") != std::string::npos) {
      snprintf(tfl, sizeof(tfl), "CHECKSUM-FAIL");
      snprintf(vs, sizeof(vs), "—"); snprintf(tile, sizeof(tile), "%dx%d", r.tileM, r.tileN);
    } else {
      snprintf(tfl, sizeof(tfl), "—"); snprintf(vs, sizeof(vs), "—"); snprintf(tile, sizeof(tile), "—");
    }
    if (sw.r && sw.r->ok) {
      snprintf(swt, sizeof(swt), "%.3f", sw.r->tflops);
      if (r.ok) snprintf(ratio, sizeof(ratio), "%.2fx", r.tflops / sw.r->tflops);
      else snprintf(ratio, sizeof(ratio), "—");
    } else {
      snprintf(swt, sizeof(swt), "—"); snprintf(ratio, sizeof(ratio), "—");
    }
    printf("%-12s %-17s %-9s %-8s %-26s %-13s %s\n", f.key, tfl, vs, tile, sw.name, swt, ratio);
  }
  printf("\n");

  // --- interpretation ---
  const double kThreshFP16 = 0.8, kThreshSW = 3.0, kThreshSWLow = 1.5;
  printf("== Interpretation ==\n");
  printf("thresholds: NAX-CLASS iff matmul2d >= %.2fx fp16 matmul2d AND >= %.1fx mapped SW baseline;\n",
         kThreshFP16, kThreshSW);
  printf("            SW-CLASS iff matmul2d < %.1fx SW baseline (tracks software);\n", kThreshSWLow);
  printf("            otherwise INCONCLUSIVE. Raw numbers above are the evidence — override freely.\n");
  for (size_t fi = 0; fi < formats.size(); ++fi) {
    const FormatSpec &f = formats[fi];
    if (f.kind == K_HALF) {
      printf("%-10s CONTROL (not classified)\n", f.key);
      continue;
    }
    const BenchResult &r = mmr[fi];
    SWMap sw = swFor(f.kind);
    if (!r.ok || !sw.r || !sw.r->ok || fp16T <= 0) {
      printf("%-10s — (insufficient data)\n", f.key);
      continue;
    }
    double rFP = r.tflops / fp16T, rSW = r.tflops / sw.r->tflops;
    const char *label;
    if (rFP >= kThreshFP16 && rSW >= kThreshSW) label = "NAX-CLASS";
    else if (rSW < kThreshSWLow) label = "SW-CLASS";
    else label = "INCONCLUSIVE";
    printf("%-10s %-12s (%.2fx fp16 matmul2d, %.2fx SW %s)\n", f.key, label, rFP, rSW, sw.name);
  }
  printf("\n");

  bool harnessOK = mmr[0].ok && mmr[1].ok && swFP16.ok;
  printf("bench controls (fp16/int8 matmul2d + fp16 SW baseline): %s\n", harnessOK ? "PASS" : "FAIL");
  if (!harnessOK) {
    printf("HARNESS FAILURE: a bench control did not run correctly\n");
    return 1;
  }
  printf("harness: OK\n");
  return 0;
}

int main(int argc, char **argv) {
  @autoreleasepool {
    BOOL benchMode = NO;
    for (int i = 1; i < argc; ++i)
      if (strcmp(argv[i], "--bench") == 0) benchMode = YES;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
      fprintf(stderr, "HARNESS FAILURE: MTLCreateSystemDefaultDevice returned nil\n");
      return 2;
    }
    if (benchMode) return runBench(device);

    printf("== Device ==\n");
    printf("name: %s\n", device.name.UTF8String);
    bool metal4 = [device supportsFamily:MTLGPUFamilyMetal4];
    printf("supportsFamily MTLGPUFamilyMetal4: %s\n", metal4 ? "yes" : "no");
    bool isM5 = [device.name containsString:@"M5"];
    printf("name contains \"M5\" (g17 / NAX hint): %s\n", isM5 ? "yes" : "no");
    if (!isM5)
      printf("WARNING: expected an M5-family device; NAX results may not apply\n");
    if (!metal4)
      printf("WARNING: device does not report MTLGPUFamilyMetal4\n");
    printf("\n");

    std::vector<LangVer> langs = buildLangVersions();
    printf("== Language versions probed (newest first) ==\n");
    for (auto &lv : langs) printf("  %s\n", lv.label);
    printf("\n");

    // --- macro summary block --------------------------------------------
    static const char *kAllMacros[] = {
#if PROBE_MACRO_COUNT > 0
      PRB_MACRONAME(0),
#endif
#if PROBE_MACRO_COUNT > 1
      PRB_MACRONAME(1),
#endif
#if PROBE_MACRO_COUNT > 2
      PRB_MACRONAME(2),
#endif
#if PROBE_MACRO_COUNT > 3
      PRB_MACRONAME(3),
#endif
#if PROBE_MACRO_COUNT > 4
      PRB_MACRONAME(4),
#endif
#if PROBE_MACRO_COUNT > 5
      PRB_MACRONAME(5),
#endif
#if PROBE_MACRO_COUNT > 6
      PRB_MACRONAME(6),
#endif
#if PROBE_MACRO_COUNT > 7
      PRB_MACRONAME(7),
#endif
#if PROBE_MACRO_COUNT > 8
      PRB_MACRONAME(8),
#endif
#if PROBE_MACRO_COUNT > 9
      PRB_MACRONAME(9),
#endif
#if PROBE_MACRO_COUNT > 10
      PRB_MACRONAME(10),
#endif
#if PROBE_MACRO_COUNT > 11
      PRB_MACRONAME(11),
#endif
#if PROBE_MACRO_COUNT > 12
      PRB_MACRONAME(12),
#endif
#if PROBE_MACRO_COUNT > 13
      PRB_MACRONAME(13),
#endif
#if PROBE_MACRO_COUNT > 14
      PRB_MACRONAME(14),
#endif
#if PROBE_MACRO_COUNT > 15
      PRB_MACRONAME(15),
#endif
    };

    printf("== Compiler macro summary (value per language version) ==\n");
    printf("%-38s", "macro");
    for (auto &lv : langs) printf(" %-14s", lv.label);
    printf("\n");
    for (size_t li = 0; li < langs.size(); ++li) {
      // __METAL_VERSION__ dump per language version
      const MacroInfo &mv = macroInfo(device, "__METAL_VERSION__", li, langs);
      (void)mv;
    }
    {
      std::string row = "__METAL_VERSION__";
      printf("%-38s", row.c_str());
      for (size_t li = 0; li < langs.size(); ++li)
        printf(" %-14s", macroInfo(device, "__METAL_VERSION__", li, langs).value.c_str());
      printf("\n");
    }
    for (size_t i = 0; i < sizeof(kAllMacros) / sizeof(kAllMacros[0]); ++i) {
      printf("%-38s", kAllMacros[i]);
      for (size_t li = 0; li < langs.size(); ++li)
        printf(" %-14s", macroInfo(device, kAllMacros[i], li, langs).value.c_str());
      printf("\n");
    }
    printf("\n");

    // --- per-format ladder ------------------------------------------------
    std::vector<FormatSpec> formats = buildFormatTable();
    std::vector<Verdict> verdicts(formats.size());

    for (size_t fi = 0; fi < formats.size(); ++fi) {
      const FormatSpec &f = formats[fi];
      Verdict &vd = verdicts[fi];
      printf("== %s ==\n", f.key);
      if (!f.typeName || !f.macroName) {
        vd.macroS = "header-absent";
        printf("  MACRO: header-absent (no -D from SDK scan); later stages skipped\n\n");
        continue;
      }
      printf("  type: %s   macro: %s\n", f.typeName, f.macroName);
      if (f.kind == K_FP8E4M3 || f.kind == K_FP8E5M2) {
        const std::vector<uint8_t> &codes = fp8Codes(f.kind);
        printf("  test encodings: %zu of 256 codes (finite quarters in [-8,8]; NaN/inf encodings excluded)\n",
               codes.size());
      }

      // MACRO
      int chosen = -1;
      std::string detail;
      for (size_t li = 0; li < langs.size(); ++li) {
        const MacroInfo &mi = macroInfo(device, f.macroName, li, langs);
        if (li) detail += " ";
        detail += langs[li].label;
        detail += ":";
        detail += mi.defined ? (mi.available ? mi.value : std::string("=") + mi.value) : "undef";
        if (mi.available && chosen < 0) chosen = (int)li;
      }
      if (chosen < 0) {
        vd.macroS = "FAIL (" + detail + ")";
        printf("  MACRO: FAIL — %s undefined/zero in all probed versions (%s)\n\n",
               f.macroName, detail.c_str());
        continue;
      }
      vd.macroOK = true;
      vd.macroS = std::string("PASS (") + langs[chosen].label + ")";
      printf("  MACRO: PASS — defined in %s (using %s)\n", detail.c_str(), langs[chosen].label);

      MTLCompileOptions *opt = [MTLCompileOptions new];
      if (!langs[chosen].isDefault) opt.languageVersion = (MTLLanguageVersion)langs[chosen].value;

      // COMPILE
      NSError *err = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:kernelSource(f.typeName)
                                                options:opt
                                                  error:&err];
      if (!lib) {
        vd.compileS = std::string("FAIL (") + firstDiagLine(err).UTF8String + ")";
        printf("  COMPILE: FAIL — %s\n\n", err.localizedDescription.UTF8String);
        continue;
      }
      vd.compileOK = true;
      vd.compileS = std::string("PASS (") + langs[chosen].label + ")";
      printf("  COMPILE: PASS (%s)\n", langs[chosen].label);

      // PIPELINE
      id<MTLFunction> fn = [lib newFunctionWithName:@"nax_probe_kernel"];
      if (!fn) {
        vd.pipelineS = "FAIL (function missing)";
        printf("  PIPELINE: FAIL — nax_probe_kernel not in library\n\n");
        continue;
      }
      err = nil;
      id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:&err];
      if (!pso) {
        vd.pipelineS = std::string("FAIL (") + firstDiagLine(err).UTF8String + ")";
        printf("  PIPELINE: FAIL — %s\n\n", err ? err.localizedDescription.UTF8String : "(unknown)");
        continue;
      }
      vd.pipelineOK = true;
      vd.pipelineS = "PASS";
      printf("  PIPELINE: PASS (threadExecutionWidth %lu)\n", (unsigned long)pso.threadExecutionWidth);

      // PARITY
      bool subByte = (f.kind == K_UINT4 || f.kind == K_INT4 || f.kind == K_FP4 || f.kind == K_INT2);
      bool isFp8 = (f.kind == K_FP8E4M3 || f.kind == K_FP8E5M2);
      int orders[2] = {0, 1};
      int nOrders = subByte ? 2 : 1;
      double bestErr = -1.0;
      int passOrder = -1;
      std::string runErr;
      for (int oi = 0; oi < nOrders; ++oi) {
        double e = parityCheck(device, pso, f, orders[oi], runErr);
        if (e < 0) {
          vd.parityS = std::string("FAIL (dispatch: ") + runErr + ")";
          printf("  PARITY: FAIL — %s\n\n", runErr.c_str());
          break;
        }
        printf("  PARITY: max abs err %.6g (pack order %s, tol %.3g)\n", e,
               subByte ? packOrderName(f.kind, orders[oi]) : (isFp8 ? "byte" : "n/a"), f.tol);
        if (bestErr < 0 || e < bestErr) bestErr = e;
        if (e <= f.tol) { passOrder = orders[oi]; break; }
      }
      printf("\n");
      if (passOrder >= 0) {
        vd.parityOK = true;
        char buf[128];
        if (subByte)
          snprintf(buf, sizeof(buf), "PASS (err %.3g, %s)", bestErr, packOrderName(f.kind, passOrder));
        else if (isFp8)
          snprintf(buf, sizeof(buf), "PASS (err %.3g, byte)", bestErr);
        else
          snprintf(buf, sizeof(buf), "PASS (err %.3g)", bestErr);
        vd.parityS = buf;
      } else if (vd.parityS == "—") {
        char buf[160];
        if (subByte)
          snprintf(buf, sizeof(buf), "FAIL (err %.3g, both orders)", bestErr);
        else
          snprintf(buf, sizeof(buf), "FAIL (err %.3g)", bestErr);
        vd.parityS = buf;
      }
    }

    // --- verdict matrix ----------------------------------------------------
    printf("== Verdict matrix ==\n");
    printf("%-16s %-24s %-22s %-12s %s\n", "FORMAT", "MACRO", "COMPILE", "PIPELINE", "PARITY");
    for (size_t fi = 0; fi < formats.size(); ++fi) {
      printf("%-16s %-24s %-22s %-12s %s\n", formats[fi].key, verdicts[fi].macroS.c_str(),
             verdicts[fi].compileS.c_str(), verdicts[fi].pipelineS.c_str(),
             verdicts[fi].parityS.c_str());
    }
    printf("\n");

    bool controlsOK = verdicts[0].macroOK && verdicts[0].compileOK && verdicts[0].pipelineOK &&
                      verdicts[0].parityOK && verdicts[1].macroOK && verdicts[1].compileOK &&
                      verdicts[1].pipelineOK && verdicts[1].parityOK;
    printf("controls (fp16, int8): %s\n", controlsOK ? "PASS" : "FAIL");
    if (!controlsOK) {
      printf("HARNESS FAILURE: a control format did not pass all four stages\n");
      return 1;
    }
    printf("harness: OK\n");
    return 0;
  }
}
