// Inkling 2's loop message gate, first half: the rank-R projection of the RMS-normed pair, one threadgroup an
// output so no thread walks more than a few inputs.
//
// hidden [rows, H] (any float); messages [rows, NQ, H] f32, of which row sel[0] is this gate's; down [2H, R] in
// TD; st [rows, R] f32 = TD(TD(rms(hidden) | rms(message)) @ down): the normed pair and the projection are rounded
// to TD, as the op chain's cast and matmul round them.
const uint tg = threadgroup_position_in_grid.x;
const uint row = tg / R;
const uint r = tg % R;
const uint tid = thread_position_in_threadgroup.x;
const uint lane = thread_index_in_simdgroup;
const uint sg = simdgroup_index_in_threadgroup;
const size_t hb = size_t(row) * H;
const size_t mb = (size_t(row) * NQ + sel[0]) * H;
threadgroup float red[TG / 32];

float sh = 0.0f;
float sm = 0.0f;
for (int d = tid; d < H; d += TG) {
  const float a = float(hidden[hb + d]);
  const float b = messages[mb + d];
  sh += a * a;
  sm += b * b;
}
const float ih = rsqrt(block_sum<TG>(sh, red, lane, sg) / H + 1e-6f);
const float im = rsqrt(block_sum<TG>(sm, red, lane, sg) / H + 1e-6f);

float acc = 0.0f;
for (int d = tid; d < H; d += TG) {
  acc += float(TD(float(hidden[hb + d]) * ih)) * float(down[size_t(d) * R + r]);
  acc += float(TD(messages[mb + d] * im)) * float(down[size_t(H + d) * R + r]);
}
const float total = block_sum<TG>(acc, red, lane, sg);
if (tid == 0) st[size_t(row) * R + r] = float(TD(total));
