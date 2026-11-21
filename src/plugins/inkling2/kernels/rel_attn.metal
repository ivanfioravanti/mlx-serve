// Causal flash attention for a prefill chunk with a relative-position bias and one sink logit per query row.
// Port of mlx-lm's `relative_attention` (mlx_lm/models/inkling2/kernels.py, Copyright (c) 2026 Apple Inc., MIT),
// with the key axis cut into passes that carry each row's max, sum and unnormalized output in fp32.
//
// q [B,HQ,L,D], k and v [B,HKV,S,D] (strided views; the queries are the last L keys), rel [B,L,HQ,R],
// sinks [B,L,HQ] f32, key_start [B], sizes {L, S, first key, end key, 1 = takes a carry | 2 = last pass}.
// Key j is visible to row i at d = S - L + i - j when 0 <= d, j >= key_start and (WINDOW == 0 or d < WINDOW);
// its logit is q.k * scale + (d < R ? rel[d] : 0). The sink is one more softmax column whose value is zero.
// q, k and v are staged as half: every finite input must lie within half's range.
constexpr float LOG2E = 1.4426950408889634f;
constexpr int BK = 16;
constexpr int LDK = BK + 8;
constexpr int LDV = D + 8;
const int L = sizes[0];
const int S = sizes[1];
const int k_begin = sizes[2];
const int k_end = sizes[3];
const bool has_carry = (sizes[4] & 1) != 0;
const bool is_final = (sizes[4] & 2) != 0;
const uint tid = thread_position_in_threadgroup.x;
const uint sg = simdgroup_index_in_threadgroup;
const ushort lane = ushort(thread_index_in_simdgroup);
const int batch = threadgroup_position_in_grid.z;
const int head = threadgroup_position_in_grid.y;
const int kv_head = head / (HQ / HKV);
const int q0 = threadgroup_position_in_grid.x * BQ;
const int start = key_start[batch];
const float qscale = scale[0] * LOG2E;

// The lane holds row fm and columns fn, fn + 1 of each 8x8 fragment.
const short2 co = msv_coord(lane);
const short fn = co.x;
const short fm = co.y;
const int row = q0 + int(sg) * 8 + fm;
const bool row_ok = row < L;
const int diag = S - L + row;
const size_t crow = size_t(batch * L + (row_ok ? row : 0)) * HQ + head;

threadgroup half Ks[D * LDK];
threadgroup half Vs[BK * LDV];

simdgroup_half8x8 q_frag[D / 8];
for (int db = 0; db < D / 8; ++db) {
  half2 qv = half2(0.0h);
  if (row_ok) {
    const device T* qp = q + batch * q_strides[0] + head * q_strides[1] + row * q_strides[2] + db * 8 + fn;
    qv = half2(float(qp[0]), float(qp[1]));
  }
  q_frag[db].thread_elements()[0] = qv.x;
  q_frag[db].thread_elements()[1] = qv.y;
}
simdgroup_float8x8 acc[D / 8];
// The row's softmax starts from its sink: exp2(0) = 1 in the sum, nothing in the output.
float row_max = 0.0f;
float row_sum = 1.0f;
if (has_carry && row_ok) {
  row_max = m_in[crow];
  row_sum = l_in[crow];
  for (int db = 0; db < D / 8; ++db) {
    acc[db].thread_elements()[0] = o_in[crow * D + db * 8 + fn];
    acc[db].thread_elements()[1] = o_in[crow * D + db * 8 + fn + 1];
  }
} else {
  if (row_ok) row_max = float(sinks[crow]) * LOG2E;
  for (int db = 0; db < D / 8; ++db) {
    acc[db].thread_elements()[0] = 0.0f;
    acc[db].thread_elements()[1] = 0.0f;
  }
}
const size_t bias = crow * R;

const int first_diag = S - L + q0;
const int last_diag = min(S - L + min(q0 + BQ, L) - 1, k_end - 1);
int first_key = WINDOW > 0 ? max(0, first_diag - WINDOW + 1) : 0;
first_key = max(first_key, max(start, k_begin));
const device T* k_head = k + batch * k_strides[0] + kv_head * k_strides[1];
const device T* v_head = v + batch * v_strides[0] + kv_head * v_strides[1];

for (int k0 = first_key / BK * BK; k0 <= last_diag; k0 += BK) {
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int c = tid; c < BK * D / 8; c += TG) {
    const int r = c / (D / 8);
    const int col = (c % (D / 8)) * 8;
    const int key = k0 + r;
    for (int j = 0; j < 8; ++j) {
      half kx = 0.0h;
      half vx = 0.0h;
      if (key < S) {
        kx = half(float(k_head[key * k_strides[2] + col + j]));
        vx = half(float(v_head[key * v_strides[2] + col + j]));
      }
      Ks[(col + j) * LDK + r] = kx;
      Vs[r * LDV + col + j] = vx;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (q0 + int(sg) * 8 >= L) continue;

  simdgroup_float8x8 scores[BK / 8];
  for (int kb = 0; kb < BK / 8; ++kb) {
    scores[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    for (int db = 0; db < D / 8; ++db) {
      simdgroup_half8x8 k_frag;
      simdgroup_load(k_frag, Ks + db * 8 * LDK + kb * 8, LDK);
      simdgroup_multiply_accumulate(scores[kb], q_frag[db], k_frag, scores[kb]);
    }
  }

  float tile_max = -INFINITY;
  for (int kb = 0; kb < BK / 8; ++kb) {
    for (int e = 0; e < 2; ++e) {
      const int key = k0 + kb * 8 + fn + e;
      const int d = diag - key;
      const bool visible = row_ok && d >= 0 && key >= start && (WINDOW == 0 || d < WINDOW);
      float s = -INFINITY;
      if (visible) {
        s = scores[kb].thread_elements()[e] * qscale;
        if (d < R) s += float(rel[bias + d]) * LOG2E;
      }
      scores[kb].thread_elements()[e] = s;
      tile_max = max(tile_max, s);
    }
  }
  // The 4 lanes of a row are lane ^ 1 and lane ^ 8 apart.
  tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(1)));
  tile_max = max(tile_max, simd_shuffle_xor(tile_max, ushort(8)));
  const float new_max = max(row_max, tile_max);
  const float factor = fast::exp2(row_max - new_max);
  float tile_sum = 0.0f;
  simdgroup_half8x8 probs[BK / 8];
  for (int kb = 0; kb < BK / 8; ++kb) {
    for (int e = 0; e < 2; ++e) {
      const float p = fast::exp2(scores[kb].thread_elements()[e] - new_max);
      probs[kb].thread_elements()[e] = half(p);
      tile_sum += p;
    }
  }
  tile_sum += simd_shuffle_xor(tile_sum, ushort(1));
  tile_sum += simd_shuffle_xor(tile_sum, ushort(8));
  row_sum = row_sum * factor + tile_sum;
  row_max = new_max;
  for (int db = 0; db < D / 8; ++db) {
    acc[db].thread_elements()[0] *= factor;
    acc[db].thread_elements()[1] *= factor;
    for (int kb = 0; kb < BK / 8; ++kb) {
      simdgroup_half8x8 v_frag;
      simdgroup_load(v_frag, Vs + kb * 8 * LDV + db * 8, LDV);
      simdgroup_multiply_accumulate(acc[db], probs[kb], v_frag, acc[db]);
    }
  }
}

if (row_ok) {
  if (is_final) {
    for (int db = 0; db < D / 8; ++db) {
      out[crow * D + db * 8 + fn] = T(acc[db].thread_elements()[0] / row_sum);
      out[crow * D + db * 8 + fn + 1] = T(acc[db].thread_elements()[1] / row_sum);
    }
  } else {
    m_out[crow] = row_max;
    l_out[crow] = row_sum;
    for (int db = 0; db < D / 8; ++db) {
      o_out[crow * D + db * 8 + fn] = acc[db].thread_elements()[0];
      o_out[crow * D + db * 8 + fn + 1] = acc[db].thread_elements()[1];
    }
  }
}
