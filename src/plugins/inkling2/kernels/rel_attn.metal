// Causal flash attention for a prefill chunk with a relative-position bias and one sink logit per query row.
// The arithmetic is mlx-lm's `relative_attention` (mlx_lm/models/inkling2/kernels.py, Copyright (c) 2026 Apple
// Inc., MIT); the tiles are MLX steel attention's as `kernels/attention/attn256.metal` lays them out: fp32
// fragments, Q in registers, K and V staged as they are stored. The key axis is cut into passes that carry each
// row's max, sum and unnormalized output in fp32.
//
// q [B,HQ,L,D]; k, v [B,HKV,S,D] (strided views whose last L rows are the queries' own); rel [B,L,HQ,R];
// sinks [B,L,HQ] f32; key_start [B]; sizes {L, S, first key, end key, 1 = takes a carry | 2 = last pass}.
// Row i sits at key S - L + i and sees key j at d = that - j when 0 <= d, j >= key_start and (WINDOW == 0 or
// d < WINDOW); the logit is q.k * scale + (d < R ? rel[d] : 0). The sink is one more softmax column whose value
// is zero: it seeds the row's max and sum.
constexpr float LOG2E = 1.4426950408889634f;
// Keys a block stages; `rel_attn.zig` cuts passes at multiples of it.
constexpr int BK = 16;
constexpr int LDV = D + 8;
const int L = sizes[0];
const int S = sizes[1];
const int k_begin = sizes[2];
const int k_end = sizes[3];
const bool has_carry = (sizes[4] & 1) != 0;
const bool is_final = (sizes[4] & 2) != 0;
const int tix = int(thread_index_in_threadgroup);
const int warp = int(simdgroup_index_in_threadgroup);
const ushort lane = ushort(thread_index_in_simdgroup);
const int batch = int(threadgroup_position_in_grid.z);
const int head = int(threadgroup_position_in_grid.y);
const int q0 = int(threadgroup_position_in_grid.x) * BQ;
const int start = key_start[batch];
const float qscale = scale[0] * LOG2E;

const short2 sc = msv_coord(lane);
const short sn = sc.x;
const short sm = sc.y;
const int tm = 8 * warp;
const int row = q0 + tm + sm;
const bool row_ok = row < L;
const int pos = S - L + row;
const size_t crow = size_t(batch * L + (row_ok ? row : 0)) * HQ + head;
const size_t bias = crow * R;

const device T* Kp = k + batch * k_strides[0] + (head / (HQ / HKV)) * k_strides[1];
const device T* Vp = v + batch * v_strides[0] + (head / (HQ / HKV)) * v_strides[1];

threadgroup T Ks[BK * LDV];
threadgroup T Vs[BK * LDV];
const int Vs_off = sm * LDV + sn;

float2 Qfrag[D / 8];
if (row_ok) {
  const device T* Qrow = q + batch * q_strides[0] + head * q_strides[1] + (long)row * q_strides[2];
  for (int dd = 0; dd < D / 8; ++dd) {
    Qfrag[dd] = float2(float(Qrow[dd * 8 + sn]), float(Qrow[dd * 8 + sn + 1]));
  }
} else {
  for (int dd = 0; dd < D / 8; ++dd) Qfrag[dd] = float2(0.0f);
}

float2 Ofrag[D / 8];
float max_score = 0.0f;
float sum_score = 1.0f;
if (has_carry && row_ok) {
  max_score = m_in[crow];
  sum_score = l_in[crow];
  for (int dd = 0; dd < D / 8; ++dd) Ofrag[dd] = float2(o_in[crow * D + dd * 8 + sn], o_in[crow * D + dd * 8 + sn + 1]);
} else {
  if (row_ok) max_score = float(sinks[crow]) * LOG2E;
  for (int dd = 0; dd < D / 8; ++dd) Ofrag[dd] = float2(0.0f);
}

const int q_lo = S - L + q0;
const int q_hi = q_lo + BQ - 1;
const int kb_lim = min((min(k_end, S) + BK - 1) / BK, (q_hi + BK) / BK);
int kb = max(k_begin, start) / BK;
if (WINDOW > 0) kb = max(kb, max(0, q_lo - WINDOW + 1) / BK);
const int kb_min_causal = max(0, q_lo) / BK;

for (; kb < kb_lim; kb++) {
  const int c0 = kb * BK;
  const int rows_k = min(BK, S - c0);

  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = tix; i < BK * (D / 8); i += TG) {
    const int r = i / (D / 8);
    const int c8 = i % (D / 8);
    uint4 wk = uint4(0);
    uint4 wv = uint4(0);
    if (r < rows_k) {
      wk = *((const device uint4*)(Kp + (long)(c0 + r) * k_strides[2]) + c8);
      wv = *((const device uint4*)(Vp + (long)(c0 + r) * v_strides[2]) + c8);
    }
    *((threadgroup uint4*)(Ks + r * LDV) + c8) = wk;
    *((threadgroup uint4*)(Vs + r * LDV) + c8) = wv;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float2 Sfrag[BK / 8];
  for (int i = 0; i < BK / 8; ++i) Sfrag[i] = float2(0.0f);
  for (int dd = 0; dd < D / 8; ++dd) {
    const float2 qf = Qfrag[dd];
    const int kbase = sn * LDV + dd * 8 + sm;
    for (int kt = 0; kt < BK / 8; ++kt) {
      const float2 kf = float2(float(Ks[kbase + kt * 8 * LDV]), float(Ks[kbase + kt * 8 * LDV + LDV]));
      msv_mma(Sfrag[kt], qf, kf);
    }
  }
  for (int kt = 0; kt < BK / 8; ++kt) Sfrag[kt] *= qscale;

  // Only blocks within R keys of the tile carry a bias, and only edge blocks hide a key: the ragged key tail,
  // the causal diagonal, the window's far side, the row's first key.
  const bool near = q_lo - (c0 + BK - 1) < R;
  const bool edge = rows_k < BK || kb >= kb_min_causal || c0 < start || (WINDOW > 0 && c0 <= q_hi - WINDOW);
  if (near || edge) {
    for (int kt = 0; kt < BK / 8; ++kt) {
      for (int jj = 0; jj < 2; ++jj) {
        const int key = c0 + kt * 8 + sn + jj;
        const int d = pos - key;
        const bool visible = row_ok && d >= 0 && key >= start && (WINDOW == 0 || d < WINDOW);
        if (!visible) {
          Sfrag[kt][jj] = -INFINITY;
        } else if (d < R) {
          Sfrag[kt][jj] += float(rel[bias + d]) * LOG2E;
        }
      }
    }
  }

  float new_max = max_score;
  for (int kt = 0; kt < BK / 8; ++kt) new_max = max(new_max, msv_row_max(Sfrag[kt]));
  float rowsum = 0.0f;
  for (int kt = 0; kt < BK / 8; ++kt) {
    Sfrag[kt] = fast::exp2(Sfrag[kt] - new_max);
    rowsum += msv_row_sum(Sfrag[kt]);
  }
  const float factor = fast::exp2(max_score - new_max);
  max_score = new_max;
  sum_score = sum_score * factor + rowsum;
  for (int i = 0; i < D / 8; ++i) Ofrag[i] *= factor;

  for (int id = 0; id < D / 8; ++id) {
    const int vbase = Vs_off + id * 8;
    for (int kt = 0; kt < BK / 8; ++kt) {
      const float2 vf = float2(float(Vs[vbase + kt * 8 * LDV]), float(Vs[vbase + kt * 8 * LDV + 1]));
      msv_mma(Ofrag[id], Sfrag[kt], vf);
    }
  }
}

if (row_ok) {
  if (is_final) {
    const float inv = 1.0f / sum_score;
    for (int id = 0; id < D / 8; ++id) {
      out[crow * D + id * 8 + sn] = T(Ofrag[id].x * inv);
      out[crow * D + id * 8 + sn + 1] = T(Ofrag[id].y * inv);
    }
  } else {
    m_out[crow] = max_score;
    l_out[crow] = sum_score;
    for (int id = 0; id < D / 8; ++id) {
      o_out[crow * D + id * 8 + sn] = Ofrag[id].x;
      o_out[crow * D + id * 8 + sn + 1] = Ofrag[id].y;
    }
  }
}
