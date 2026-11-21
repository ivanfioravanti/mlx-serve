// The key-split relative attention's second half: each row's chunk states (max and sum in log2 units, the
// unnormalized output) joined in chunk order and normalized, one simdgroup a row.
//
// ms, ls [CH, rows] f32; os [CH, rows, D] f32; out [rows, D] in T. A chunk that saw no key holds a sum of 0 and
// weighs nothing.
const uint row = thread_position_in_grid.x / 32;
const uint lane = thread_index_in_simdgroup;
const size_t rows = threadgroups_per_grid.x;
float top = -INFINITY;
for (int c = 0; c < CH; c++) top = max(top, ms[c * rows + row]);
float w[CH];
float total = 0.0f;
for (int c = 0; c < CH; c++) {
  w[c] = fast::exp2(ms[c * rows + row] - top);
  total += ls[c * rows + row] * w[c];
}
for (int d = lane; d < D; d += 32) {
  float acc = 0.0f;
  for (int c = 0; c < CH; c++) acc += os[(c * rows + row) * D + d] * w[c];
  out[size_t(row) * D + d] = T(acc / total);
}
