// Inkling 2: J streams of S rows, Y [J, S, D] f32, under their weighted RMS norms (weights NW0..NW2 [D], eps PR[0]).
// grid (D, S), threadgroup (TG, 1): one thread per stream element of a row; sums as pre.metal's. X [J, S, D] T.
const uint t = thread_position_in_threadgroup.x;
const uint e = thread_position_in_grid.x;
const uint row = thread_position_in_grid.y;
const uint S = Y_shape[1];
const uint lane = thread_index_in_simdgroup;
const uint sg = t / 32;
constexpr uint NSG = TG / 32;
threadgroup float red[NSG * J];
// The streams' folds of the SIMD groups' partial sums, each by one thread in the order every thread used to run it.
threadgroup float fold[J];
float ss[J], mine[J];
for (uint j = 0; j < J; j++) ss[j] = 0.0f;
for (uint i = t; i < D; i += TG) for (uint j = 0; j < J; j++) {
    const float y = Y[(j * S + row) * D + i];
    ss[j] += y * y;
    if (i == e) mine[j] = y;
}
for (uint j = 0; j < J; j++) {
    ss[j] = simd_sum(ss[j]);
    if (lane == 0) red[sg * J + j] = ss[j];
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = t; j < J; j += TG) {
    float s = 0.0f;
    for (uint g = 0; g < NSG; g++) s += red[g * J + j];
    fold[j] = s;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint j = 0; j < J; j++) ss[j] = metal::precise::rsqrt(fold[j] / float(D) + PR[0]);
for (uint j = 0; j < J; j++) {
    const float nw = float(j == 0 ? NW0[e] : (j == 1 ? NW1[e] : NW2[e]));
    X[(j * S + row) * D + e] = T(mine[j] * ss[j] * nw);
}
