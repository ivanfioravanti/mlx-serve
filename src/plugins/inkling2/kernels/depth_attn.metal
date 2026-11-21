// Depth attention: learned queries attend over the depth sources of each token (up to 8 block states, then the
// partial one), which are also the values. Port of mlx-lm's `depth_attention`
// (mlx_lm/models/inkling2/kernels.py, Copyright (c) 2026 Apple Inc., MIT) with the block states as separate inputs,
// so the bank is never stacked.
//
// b0..b7 and partial [rows, D] f32 (b_i is read only for i < K); queries [Q, D] f32; scale [1]; out [rows, Q, D].
// One threadgroup a row: each thread sums every TG-th product, the lanes and then the simdgroups are summed, the
// softmax over the K + 1 sources runs in every thread, and each thread writes its own columns.
constexpr int N = K + 1;
const uint row = threadgroup_position_in_grid.x;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;

float acc[N * Q];
for (int i = 0; i < N * Q; i++) acc[i] = 0.0f;
for (int d = tid; d < D; d += TG) {
  const size_t at = size_t(row) * D + d;
  float x[N];
  if (K > 0) x[0] = b0[at];
  if (K > 1) x[1] = b1[at];
  if (K > 2) x[2] = b2[at];
  if (K > 3) x[3] = b3[at];
  if (K > 4) x[4] = b4[at];
  if (K > 5) x[5] = b5[at];
  if (K > 6) x[6] = b6[at];
  if (K > 7) x[7] = b7[at];
  x[K] = partial[at];
  for (int j = 0; j < Q; j++) {
    const float qv = queries[j * D + d];
    for (int i = 0; i < N; i++) acc[i * Q + j] += x[i] * qv;
  }
}

threadgroup float buf[N * Q][TG / 32];
for (int i = 0; i < N * Q; i++) {
  const float v = simd_sum(acc[i]);
  if (lane == 0) buf[i][sg] = v;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

float w[N * Q];
for (int i = 0; i < N * Q; i++) {
  float t = 0.0f;
  for (int s = 0; s < TG / 32; s++) t += buf[i][s];
  w[i] = t * scale[0];
}
for (int j = 0; j < Q; j++) {
  float m = -INFINITY;
  for (int i = 0; i < N; i++) m = max(m, w[i * Q + j]);
  float z = 0.0f;
  for (int i = 0; i < N; i++) {
    w[i * Q + j] = metal::exp(w[i * Q + j] - m);
    z += w[i * Q + j];
  }
  for (int i = 0; i < N; i++) w[i * Q + j] /= z;
}

for (int d = tid; d < D; d += TG) {
  const size_t at = size_t(row) * D + d;
  float x[N];
  if (K > 0) x[0] = b0[at];
  if (K > 1) x[1] = b1[at];
  if (K > 2) x[2] = b2[at];
  if (K > 3) x[3] = b3[at];
  if (K > 4) x[4] = b4[at];
  if (K > 5) x[5] = b5[at];
  if (K > 6) x[6] = b6[at];
  if (K > 7) x[7] = b7[at];
  x[K] = partial[at];
  for (int j = 0; j < Q; j++) {
    float o = 0.0f;
    for (int i = 0; i < N; i++) o += w[i * Q + j] * x[i];
    out[(size_t(row) * Q + j) * D + d] = o;
  }
}
