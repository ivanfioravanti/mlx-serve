// Inkling 2's short causal conv over channels (mlx-lm's `ShortConvolution` and `DynamicShortConvolution`) in one
// dispatch: the window (the K - 1 rows before the chunk, then the chunk in f32), the conv, the residual and its
// scale. One thread an output (row t, channel c); it also writes its window rows.
//
// x [1, L, C] T, or with KE > 0 the chunk's top-KE expert outputs [L, KE, C] T and their weights wts [L, KE] f32,
// whose weighted f32 sum rounded to T is the chunk (the MoE's last step); prev [1, K - 1, C] f32 (read when PREV,
// else zeros); w: static [C, K, 1] in TW (the conv runs on the window rounded to TW, its sum rounded to TW), dynamic (DYN) [K, C] plus dyn [1, L, C / HD * K]: f32 taps
// w[k][c] + dyn[t][c / HD][k] over the f32 window; scale [1] f32. out [1, L, C] T = T((y + x) * scale); win
// [1, K - 1 + L, C] f32. POST then norms each HD-wide head of the rounded output in f32 (one threadgroup a head):
// 1 = RMS (eps post[0]), 2 = L2 clip to post[0]; and rounds it to T again.
// Products round before they are summed, as the op chain's multiply and sum do.
#pragma clang fp contract(off)
#define XAT(r, v)                                                                                 \
  if (KE == 0) {                                                                                  \
    v = float(x[(r) * x_strides[1] + c * x_strides[2]]);                                          \
  } else {                                                                                        \
    float s_ = 0.0f;                                                                              \
    for (int j = 0; j < KE; j++)                                                                  \
      s_ += float(x[(r) * x_strides[0] + j * x_strides[1] + c * x_strides[2]]) * wts[(r) * wts_strides[0] + j * wts_strides[1]]; \
    v = float(T(s_));                                                                             \
  }
const uint gid = thread_position_in_grid.x;
const uint t = gid / C;
const uint c = gid % C;
float xt;
XAT(t, xt);
win[size_t(t + K - 1) * C + c] = xt;

float acc = 0.0f;
for (int i = 0; i < K; i++) {
  // The dynamic form sums by lag (lag i reads window row t + K - 1 - i), the static one by tap (tap i, row t + i).
  const int k = DYN ? K - 1 - i : i;
  const int r = int(t) + k - (K - 1);
  float v = 0.0f;
  if (r >= 0) {
    XAT(r, v);
  } else if (PREV) {
    v = prev[(r + K - 1) * prev_strides[1] + c * prev_strides[2]];
  }
  if (t == 0 && r < 0) win[size_t(r + K - 1) * C + c] = v;
  if (DYN) {
    acc += v * (float(w[i * C + c]) + float(dyn[t * dyn_strides[1] + ((c / HD) * K + i) * dyn_strides[2]]));
  } else {
    acc += float(TW(v)) * float(w[c * K + k]);
  }
}
const float y = DYN ? acc : float(TW(acc));
const T o = T((y + xt) * scale[0]);
if (POST == 0) {
  out[size_t(t) * C + c] = o;
} else {
  threadgroup float red[32];
  const uint lane = thread_index_in_simdgroup;
  const uint sg = simdgroup_index_in_threadgroup;
  const float f = float(o);
  float ss = simd_sum(f * f);
  if (lane == 0) red[sg] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  ss = 0.0f;
  for (int i = 0; i < HD / 32; i++) ss += red[i];
  float factor;
  if (POST == 1) {
    factor = metal::precise::rsqrt(ss / HD + post[0]);
  } else {
    // A NaN factor stays NaN, as MLX's `minimum` keeps it.
    const float r = post[0] / (metal::precise::sqrt(ss) + 1e-6f);
    factor = metal::isnan(r) ? r : metal::min(r, 1.0f);
  }
  out[size_t(t) * C + c] = T(f * factor);
}
