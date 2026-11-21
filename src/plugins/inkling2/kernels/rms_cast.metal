// RMS norm of each D-long row in f32 (MLX's `rms_norm`: four values a thread, the inverse root of the mean square
// plus eps), times the gain (W) and cast to OT in one dispatch: the op chain is a cast in, the norm and a cast out.
// TWIN also writes the rounded output widened to f32 (a router's input).
//
// x [rows, D] (any float); w [D] f32 (read when W); eps [1] f32; out [rows, D] OT; twin [rows, D] f32 (TWIN).
const uint row = threadgroup_position_in_grid.x;
const uint lid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint nsg = (threads_per_threadgroup.x + 31) / 32;
threadgroup float sums[32];

const size_t at = size_t(row) * D;
float xs[4];
float acc = 0.0f;
for (int i = 0; i < 4; i++) {
  const uint d = lid * 4 + i;
  xs[i] = d < D ? float(x[at + d]) : 0.0f;
  acc += xs[i] * xs[i];
}
acc = simd_sum(acc);
if (lane == 0) sums[sg] = acc;
threadgroup_barrier(mem_flags::mem_threadgroup);
if (sg == 0) {
  float v = lane < nsg ? sums[lane] : 0.0f;
  v = simd_sum(v);
  if (lane == 0) sums[0] = metal::precise::rsqrt(v / D + eps[0]);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const float inv = sums[0];
for (int i = 0; i < 4; i++) {
  const uint d = lid * 4 + i;
  if (d < D) {
    const OT y = OT((W ? w[d] : 1.0f) * (xs[i] * inv));
    out[at + d] = y;
    if (TWIN) twin[at + d] = float(y);
  }
}
