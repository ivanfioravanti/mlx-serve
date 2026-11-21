// Inkling 2's loop message gate, second half: both gates read back from the projection and mixed, one simdgroup
// an output.
//
// st [rows, R] f32; forget, gain [R, H] f32; hidden [rows, H] (any float); messages [rows, NQ, H] f32 (row sel[0]);
// out [rows, H] f32 = (1 - tanh(st.forget)) * hidden + tanh(st.gain) * message.
const uint at = thread_position_in_grid.x / 32;
const uint row = at / H;
const uint d = at % H;
const uint lane = thread_index_in_simdgroup;
float f = 0.0f;
float gn = 0.0f;
for (int k = lane; k < R; k += 32) {
  const float s = st[size_t(row) * R + k];
  f += s * forget[size_t(k) * H + d];
  gn += s * gain[size_t(k) * H + d];
}
f = simd_sum(f);
gn = simd_sum(gn);
if (lane == 0) {
  out[at] = (1.0f - metal::tanh(f)) * float(hidden[at]) + metal::tanh(gn) * messages[(size_t(row) * NQ + sel[0]) * H + d];
}
