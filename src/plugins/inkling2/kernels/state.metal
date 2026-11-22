// Inkling 2: a visit's new conv state NST [1, KT - 1, SW] f32 after S rows, one thread per element: the last KT - 1
// rows of the old state ST followed by the rows' conv inputs [KP | VP | OP | XM] (the k, v and output projections
// and the MLP's output, [1, S, .]: C, C, D and D wide). With a state per sequence (ST, NST [S / SR, KT - 1, SW]) each
// moves on by its own SR rows: SR = S one sequence, 1 a sequence a row (batched slots), else a slot's prompt span.
const uint e = thread_position_in_grid.x;
const uint SW = ST_shape[2];
const uint S = KP_shape[1];
const uint C = KP_shape[2];
const uint D = XM_shape[2];
const uint col = e % SW;
// Row `at` of [the sequence's ST | its SR new rows].
const uint SR = S / ST_shape[0];
const uint seq = e / ((KT - 1) * SW);
const uint at = SR + (e / SW) % (KT - 1);
if (at < KT - 1) {
    NST[e] = ST[(seq * (KT - 1) + at) * SW + col];
    return;
}
const uint r = seq * SR + at - (KT - 1);
if (col < C) NST[e] = float(KP[r * C + col]);
else if (col < 2 * C) NST[e] = float(VP[r * C + col - C]);
else if (col < SW - D) NST[e] = float(OP[r * D + col - 2 * C]);
else NST[e] = float(XM[r * D + col - (SW - D)]);
