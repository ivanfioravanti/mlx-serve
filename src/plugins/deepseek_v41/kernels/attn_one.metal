const uint lane = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;
const int i = threadgroup_position_in_grid.y, h = threadgroup_position_in_grid.x;
const int p0 = meta[0], K = meta[2], window = meta[3], cap = meta[4], nc = meta[5];
const int H = q_shape[1];
constexpr int V = D / 32, VP = 4;
const int pos = p0 + i;
const int lo = metal::max(0, pos - window + 1);
const int nw = pos - lo + 1;
float qv[V], acc[V];
for (int v = 0; v < V; ++v) { qv[v] = float(q[(i * H + h) * D + lane * V + v]); acc[v] = 0.0f; }
float mx = -INFINITY, den = 0.0f;
const float scale = scal[0];
const int sp = sgi / NSG, sg = sgi % NSG, chunk = (nw + K + S - 1) / S;
const int t_lo = sp * chunk, t_hi = metal::min(nw + K, t_lo + chunk);
for (int t = t_lo + sg; t < t_hi; t += NSG) {
  const device KT* row;
  if (t < nw) row = (FRESH && lo + t >= meta[6]) ? fresh + size_t(lo + t - meta[6]) * D : ring + size_t((lo + t) % cap) * D;
  else {
    const int c = topk[i * K + t - nw];
    if (c < 0 || c >= nc) continue;
    row = comp + size_t(c) * D;
  }
  float kv[V], dot = 0.0f;
  for (int v = 0; v < V; ++v) { kv[v] = float(row[lane * V + v]); dot += qv[v] * kv[v]; }
  const float s = metal::simd_sum(dot) * scale;
  const float nm = metal::max(mx, s);
  const float corr = metal::exp(mx - nm), p = metal::exp(s - nm);
  den = den * corr + p;
  for (int v = 0; v < V; ++v) acc[v] = acc[v] * corr + p * kv[v];
  mx = nm;
}
threadgroup float t_acc[S * NSG * 32 * VP];
threadgroup float t_mx[S * NSG], t_den[S * NSG];
threadgroup float t_part[S * D];
if (lane == 0) { t_mx[sgi] = mx; t_den[sgi] = den; }
// A thread's split partial this pass: split js, the channel VP * pass + k of lane jl.
const int idx = sgi * 32 + lane, js = idx / (32 * VP), jl = idx / VP % 32, jk = idx % VP;
for (int ph = 0; ph < V / VP; ++ph) {
  for (int k = 0; k < VP; ++k) t_acc[idx * VP + k] = acc[ph * VP + k];
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
  if (js < S) {
    float M = -INFINITY, w[NSG];
    for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[js * NSG + g]);
    for (int g = 0; g < NSG; ++g) w[g] = t_den[js * NSG + g] > 0.0f ? metal::exp(t_mx[js * NSG + g] - M) : 0.0f;
    float o = 0.0f;
    for (int g = 0; g < NSG; ++g) o += t_acc[((js * NSG + g) * 32 + jl) * VP + jk] * w[g];
    t_part[js * D + jl * V + ph * VP + jk] = o;
  }
  threadgroup_barrier(metal::mem_flags::mem_threadgroup);
}
if (idx < D) {
  float pm[S], pl[S];
  for (int j = 0; j < S; ++j) {
    float M = -INFINITY, l = 0.0f;
    for (int g = 0; g < NSG; ++g) M = metal::max(M, t_mx[j * NSG + g]);
    for (int g = 0; g < NSG; ++g) l += t_den[j * NSG + g] * (t_den[j * NSG + g] > 0.0f ? metal::exp(t_mx[j * NSG + g] - M) : 0.0f);
    pm[j] = M;
    pl[j] = l;
  }
  float M = sink[h];
  for (int j = 0; j < S; ++j) M = metal::max(M, pm[j]);
  float total = metal::exp(sink[h] - M), o = 0.0f;
  for (int j = 0; j < S; ++j) {
    const float w = pl[j] > 0.0f ? metal::exp(pm[j] - M) : 0.0f;
    total += pl[j] * w;
    o += t_part[j * D + idx] * w;
  }
  out[(size_t(i) * H + h) * D + idx] = dsv41_unrope<OT, RD, D>(OT(o / total), idx, pos, periods);
}
