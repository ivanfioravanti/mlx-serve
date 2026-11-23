// Inkling 2, a few rows: the picked experts' down projections, their weighted sum, the MLP's short conv and the final
// add in one launch, the packed bank read in place. grid (32, KSEL, D / ROWS), threadgroup (32, KSEL, 1): a SIMD group
// owns ROWS output rows of one pick of each of the NR rows in turn (rows of WQ [E, D, I * bits / 32] u32, scales WS
// [E, D, I / GS] u8; FMT 3 an affine bank of BITS bits, scales WS and biases WB [E, D, I / GS] of S), a lane 16
// values of ACT [NR * KSEL, 1, I] (T) per sweep; the threadgroup holds every pick of those rows for every row. Each
// pick's row sum is rounded to T as the bank's matmul leaves it, times SC[IDX[p]] with SCALED (f32 [E]) and its
// routing weight WTS[p] (f32); their sum in pick order, rounded to T, is the MLP's output XM [1, NR, D]. Then sum.metal's
// tail: the conv over the rows behind the window before them (the last D columns of ST [1, KT - 1, SW] f32, or a
// state per row, ST [NR, KT - 1, SW]) with MW [D, KT, 1] (WT), its sum rounded to WT as the conv op does, plus XM,
// times PR[0], rounded to T, and the residual RES [1, NR, D] f32 added: HOUT [1, NR, D] (T), the visit's stream.
const uint lane = thread_index_in_simdgroup;
const uint e = thread_position_in_grid.y;
const uint row0 = thread_position_in_grid.z * ROWS;
constexpr uint WPR = FMT == 2 ? I / 4 : I / 8;
constexpr uint WBR = I * BITS / 8;
constexpr uint SPR = I / GS;
const uint SW = ST_shape[2];
const uint OFF = SW - D;
threadgroup float part[NR * KSEL * ROWS];
for (uint srow = 0; srow < NR; srow++) {
    const uint p = srow * KSEL + e;
    const size_t base = size_t(IDX[p]) * D + row0;
    const device uint* wrow = WQ + base * WPR;
    const device uchar* scales = (const device uchar*)WS + base * SPR;
    const device uchar* arow = (const device uchar*)WQ + base * WBR;
    const device S* sa = (const device S*)WS + base * SPR;
    const device S* ba = (const device S*)WB + base * SPR;
    float acc[ROWS];
    for (uint r = 0; r < ROWS; r++) acc[r] = 0.0f;
    for (uint k = lane * 16; k < I; k += 512) {
        float xr[16];
        for (uint i = 0; i < 16; i++) xr[i] = float(ACT[size_t(p) * I + k + i]);
        for (uint r = 0; r < ROWS; r++)
            acc[r] += FMT == 3 ? msv_ink2_affine_chunk<S, BITS, GS>(arow + r * WBR, sa + r * SPR, ba + r * SPR, k / 16, xr)
                               : msv_ink2_chunk(FMT, wrow + r * WPR, scales + r * SPR, k / 16, xr);
    }
    for (uint r = 0; r < ROWS; r++) {
        const float v = float(T(simd_sum(acc[r]))) * (SCALED ? SC[IDX[p]] : 1.0f) * WTS[p];
        if (lane == 0) part[p * ROWS + r] = v;
    }
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if (e == 0 && lane < ROWS) {
    const uint c = row0 + lane;
    T xm[NR];
    for (uint srow = 0; srow < NR; srow++) {
        float x = 0.0f;
        for (uint k = 0; k < KSEL; k++) x += part[(srow * KSEL + k) * ROWS + lane];
        xm[srow] = T(x);
        XM[size_t(srow) * D + c] = xm[srow];
    }
    // Rows a state: NR one state, 1 a state per row.
    const uint SR = NR / ST_shape[0];
    for (uint srow = 0; srow < NR; srow++) {
        const uint local = srow % SR, seq = srow / SR;
        float y = 0.0f;
        // Tap j reads row `local + j` of [the sequence's ST | its rows' MLP outputs].
        for (uint j = 0; j < KT; j++)
            y += (local + j < KT - 1 ? ST[(seq * (KT - 1) + local + j) * SW + OFF + c] : float(xm[srow + j - (KT - 1)])) * float(MW[c * KT + j]);
        y = float(WT(y));
        HOUT[size_t(srow) * D + c] = T(RES[size_t(srow) * D + c] + float(T((y + float(xm[srow])) * PR[0])));
    }
}
