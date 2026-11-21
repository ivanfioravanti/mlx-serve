// Inkling 2's routed experts for a decode row, read in place: one simdgroup an output row of a top-k slot's expert,
// its lanes striding the packed row (the SDK's `gather_qmv` shape). MODE 0 is NVFP4 (e2m1 nibbles, e4m3 scales a
// 16-value group, `GQMV_NVFP4_HEADER`'s decodes), MODE 1 MXFP8 (e4m3 bytes, e8m0 scales a 32-value group); each
// decode is a power of two off its value and one multiply restores them.
//
// PAIR (gate_up): x [K] the token, rows 2i and 2i + 1 of the bank are gate and up; each dot is rounded to T as the
// gather rounds it, scaled by gscale[expert] (f32, read when GS), and y [TOPK, N / 2] = T(cap_g tanh(g / cap_g)
// sigmoid(g) cap_u tanh(u / cap_u)) with caps [2] f32. Otherwise (down): x [TOPK, K], each slot's own activation, and
// y [TOPK, N] = T(dot). w [E, N, K * bits / 32] u32, scales [E, N, K / group] u8, inds [TOPK] u32.
const uint lane = thread_index_in_simdgroup;
const uint n = thread_position_in_grid.y;
const uint e = thread_position_in_grid.z;
const uint eid = inds[e];
constexpr int VPW = MODE == 0 ? 8 : 4;
constexpr int GSZ = MODE == 0 ? 16 : 32;
constexpr int KW = K / VPW;
constexpr int KG = K / GSZ;
const size_t xoff = PAIR ? 0 : size_t(e) * K;

float dots[2];
for (int r = 0; r < (PAIR ? 2 : 1); r++) {
  const uint row = PAIR ? 2 * n + r : n;
  const size_t wbase = (size_t(eid) * N + row) * KW;
  const size_t gbase = (size_t(eid) * N + row) * KG;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int pack = int(lane); pack < KW; pack += 32) {
    const uint packed = w[wbase + pack];
    const int k_base = pack * VPW;
    const size_t xi = xoff + size_t(k_base);
    if (MODE == 0) {
      const float sj = mlxserve_e4m3(uint(scales[gbase + k_base / GSZ]));
      const float2 p0 = mlxserve_e2m1x2(packed, 0), p1 = mlxserve_e2m1x2(packed, 1);
      const float2 p2 = mlxserve_e2m1x2(packed, 2), p3 = mlxserve_e2m1x2(packed, 3);
      a0 += (float(x[xi + 0]) * p0.x + float(x[xi + 4]) * p0.y) * sj;
      a1 += (float(x[xi + 1]) * p1.x + float(x[xi + 5]) * p1.y) * sj;
      a2 += (float(x[xi + 2]) * p2.x + float(x[xi + 6]) * p2.y) * sj;
      a3 += (float(x[xi + 3]) * p3.x + float(x[xi + 7]) * p3.y) * sj;
    } else {
      const uint sb = uint(scales[gbase + k_base / GSZ]);
      const float sj = as_type<float>(sb == 0u ? 0x400000u : (sb << 23)) * 0.00390625f;
      a0 += float(x[xi + 0]) * mlxserve_e4m3(packed & 0xFFu) * sj;
      a1 += float(x[xi + 1]) * mlxserve_e4m3((packed >> 8) & 0xFFu) * sj;
      a2 += float(x[xi + 2]) * mlxserve_e4m3((packed >> 16) & 0xFFu) * sj;
      a3 += float(x[xi + 3]) * mlxserve_e4m3(packed >> 24) * sj;
    }
  }
  // NVFP4's decodes are 2^-14 and 2^-8 off, MXFP8's 2^-8 and 2^-8.
  dots[r] = simd_sum((a0 + a1) + (a2 + a3)) * (MODE == 0 ? 4194304.0f : 65536.0f);
}
if (lane == 0) {
  if (PAIR) {
    const float s = GS ? gscale[eid] : 1.0f;
    const float g = float(T(dots[0])) * s;
    const float u = float(T(dots[1])) * s;
    const float gc = metal::precise::tanh(g / caps[0]) * caps[0];
    const float uc = metal::precise::tanh(u / caps[1]) * caps[1];
    // MLX's sigmoid.
    const float sy = 1.0f / (1.0f + metal::exp(-metal::abs(g)));
    const float sig = g < 0.0f ? 1.0f - sy : sy;
    y[size_t(e) * (N / 2) + n] = T(gc * sig * uc);
  } else {
    y[size_t(e) * N + n] = T(dots[0]);
  }
}
