// Inkling 2's short causal conv over channels (mlx-lm's `ShortConvolution` and `DynamicShortConvolution`) in one
// dispatch: the window (the K - 1 rows before the chunk, then the chunk in f32), the conv, the residual and its
// scale. One thread an output (row t, channel c); it also writes its window rows.
//
// x [1, L, C] T; prev [1, K - 1, C] f32 (read when PREV, else zeros); w: static [C, K, 1] in TW (the conv runs on the
// window rounded to TW, its sum rounded to TW), dynamic (DYN) [K, C] plus dyn [1, L, C / HD * K]: f32 taps
// w[k][c] + dyn[t][c / HD][k] over the f32 window; scale [1] f32. out [1, L, C] T = T((y + x) * scale); win
// [1, K - 1 + L, C] f32.
const uint gid = thread_position_in_grid.x;
const uint t = gid / C;
const uint c = gid % C;
const float xt = float(x[t * x_strides[1] + c * x_strides[2]]);
win[size_t(t + K - 1) * C + c] = xt;

float acc = 0.0f;
for (int i = 0; i < K; i++) {
  // The dynamic form sums by lag (lag i reads window row t + K - 1 - i), the static one by tap (tap i, row t + i).
  const int k = DYN ? K - 1 - i : i;
  const int r = int(t) + k - (K - 1);
  float v = 0.0f;
  if (r >= 0) {
    v = float(x[r * x_strides[1] + c * x_strides[2]]);
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
out[size_t(t) * C + c] = T((y + xt) * scale[0]);
