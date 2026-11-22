// Inkling 2, a prompt's S rows: the picked experts' outputs back to their rows, the MLP's short conv and the final
// add, one thread per output element. DN [N, 1, D] (TI) are the (row, pick) pairs' down projections in expert order,
// IDX [N] u32 their experts and INV [N] the place of pair (row, pick) in that order. Each is times SC[IDX] with
// SCALED (f32 [E]) and its routing weight WTS [1, S, K] (f32), summed in pick order and rounded to T: XM [1, S, D],
// the MLP's output. Then the conv over the rows behind the window before them (the last D columns of ST
// [1, KT - 1, SW] f32; or a state per sequence, ST [S / SR, KT - 1, SW], SR its rows) with MW [D, KT, 1] (WT), its sum rounded to WT as the conv op does, plus XM, times PR[0],
// rounded to T, and the residual RES [1, S, D] f32 added: HOUT [1, S, D] (T), the visit's output stream.
const uint e = thread_position_in_grid.x;
const uint D = DN_shape[2];
const uint SW = ST_shape[2];
const uint OFF = SW - D;
const uint s = e / D, c = e % D;
// Rows a sequence (`prep.metal`): SR = S one state, 1 a state per row, else a state per slot of SR rows.
const uint SR = WTS_shape[1] / ST_shape[0];
const uint local = s % SR, seq = s / SR;
float win[KT];
for (uint j = 0; j < KT; j++) {
    // Tap j reads row `local + j` of [the sequence's ST | its rows' MLP outputs].
    if (local + j < KT - 1) {
        win[j] = ST[(seq * (KT - 1) + local + j) * SW + OFF + c];
        continue;
    }
    const uint r = s + j - (KT - 1);
    float x = 0.0f;
    for (uint k = 0; k < K; k++) {
        const uint at = uint(INV[r * K + k]);
        x += float(DN[size_t(at) * D + c]) * (SCALED ? SC[IDX[at]] : 1.0f) * WTS[r * K + k];
    }
    win[j] = float(T(x));
}
XM[e] = T(win[KT - 1]);
float y = 0.0f;
for (uint j = 0; j < KT; j++) y += win[j] * float(MW[c * KT + j]);
y = float(WT(y));
HOUT[e] = T(RES[e] + float(T((y + win[KT - 1]) * PR[0])));
