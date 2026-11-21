// A sum over all TG threads of a threadgroup (mlx-lm's `BLOCK_SUM_HEADER`, mlx_lm/kernels/common.py, Copyright (c)
// 2026 Apple Inc., MIT).
template <int TG>
inline float block_sum(float v, threadgroup float* buf, uint lane, uint sg) {
  v = simd_sum(v);
  if (TG > 32) {
    if (lane == 0) buf[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = simd_sum(lane < TG / 32 ? buf[lane] : 0.0f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return v;
}
