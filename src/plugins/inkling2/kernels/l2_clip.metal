// L2 clip of each D-long vector (mlx-lm's `_l2_clip`): s = x * scale in f32, shrunk to norm `threshold` where its norm
// is past it, cast to T. One simdgroup a vector; a lane holds every 32nd value.
//
// x [V, D] (V = rows * H), scale [H, D] (read only when SCALED), threshold [1] f32; out [V, D] T.
const uint v = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
constexpr int PER = (D + 31) / 32;
const size_t at = size_t(v) * D;
const uint h = v % H;

float s[PER];
float ss = 0.0f;
for (int i = 0; i < PER; i++) {
  const int d = lane + 32 * i;
  float y = 0.0f;
  if (d < D) {
    y = float(x[at + d]);
    if (SCALED) y *= float(scale[h * D + d]);
  }
  s[i] = y;
  ss += y * y;
}
ss = simd_sum(ss);
// A NaN factor stays NaN, as MLX's `minimum` keeps it.
const float r = threshold[0] / (metal::precise::sqrt(ss) + 1e-6f);
const float f = metal::isnan(r) ? r : metal::min(r, 1.0f);
for (int i = 0; i < PER; i++) {
  const int d = lane + 32 * i;
  if (d < D) out[at + d] = T(s[i] * f);
}
