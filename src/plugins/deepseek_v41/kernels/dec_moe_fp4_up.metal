// dec_moe_fp4_up: routed (mxfp4) and shared (mxfp8) gate and up, clamped SwiGLU times the routing weight. One simdgroup
// per (output column, slot): slots 0 .. TOPK * R - 1 are the (row, pick) pairs, slot TOPK * R the shared expert (every
// row). A column's slots are adjacent in the grid, which spreads the heavy shared tiles over the launch.
// MODE 1 launches the shared slot alone (`out` [R, N]) and MODE 2 the routed slots alone (`out` [R, TOPK, N]): the same
// sums a slot, so the shared expert can run beside the router.
const uint lane = thread_index_in_simdgroup;
const uint sg = thread_position_in_grid.x / 32;
constexpr uint SLOTS = MODE == 1 ? 1u : uint(TOPK * R) + (MODE == 0 ? 1u : 0u);
constexpr uint OS = MODE == 2 ? uint(TOPK) : uint(TOPK + 1);
const uint u = sg % SLOTS;
const uint n_col = sg / SLOTS;
if (n_col >= uint(N)) return;
if (MODE == 1 || (MODE == 0 && u == uint(TOPK * R))) {
  float acc[R][2];
  dec_up_shared<R, K, N, T>(sw, sscales, n_col, x, lane, acc);
  if (lane == 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) out[MODE == 1 ? r * N + n_col : (r * (TOPK + 1) + TOPK) * N + n_col] = dec_up_act<T>(acc[r][0], acc[r][1], lim[0], 1.0f);
  }
  return;
}
const uint r = u / TOPK;
float acc[2];
dec_fp4_up_routed<K, N, T>(w, scales, size_t(inds[u]) * (2 * N) + n_col, x + size_t(r) * K, lane, acc);
if (lane == 0) out[(r * OS + u % TOPK) * N + n_col] = dec_up_act<T>(acc[0], acc[1], lim[0], wts[u]);
