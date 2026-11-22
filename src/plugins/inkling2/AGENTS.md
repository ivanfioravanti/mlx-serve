# inkling2

`inkling2_loop` (Thinking Machines Inkling 2 Mini, 26B-A5.4B), text and images, in mlx-lm's layout (PR #2086,
`mlx_lm/models/inkling2/`, the reference). Its own forward loop (`plugin.zig` `Model`), not a base: `config.zig` the
parse and the per-visit cache layout, `blocks.zig` depth attention, message gates and short convs, `attention.zig` the
attention block, `rel_attn.zig` the attention read and its simdgroup kernel, `rel_attn_nax.zig` the TensorOps arm,
`moe.zig` the routed MLP, `vision.zig` the image tower and its processor (`MediaHooks`).

- A layer VISIT is the engine's layer: `loop_config` runs layers `core_start..core_end` `num_passes` times, so the
  shell's `num_hidden_layers`, the cache layout and the `.conv` entries count visits (48 for 24 layers), and
  `c.layer` is the visit. Each visit has its own KV and its own conv windows.
- A pass after the first restarts from the PRELUDE's output and block bank; it hears the previous pass only through
  the message gates (depth attention over that pass's block states after its first, and its normed output).
- A layer opening a block (`i % n_layers_per_attn_res_block == 0`) appends its normed input to the bank and its
  residual restarts from the attention output; every other layer adds. Attention and MLP inputs are depth-attention
  reads of the bank, never the residual itself (layer 0's attention aside).
- State: a visit's four conv windows are ONE f32 array `[k | v | attn | mlp]`, written once at the layer's end.
- Attention has NO RoPE: `rel_attn.attend` takes the relative bias `[B, L, Hq, R]` and one sink logit per query row;
  a decode row goes through MLX's fused attention with the bias as its mask. Global layers scale q and the bias by
  `1 + alpha * log(position / floor)` past `log_scaling_n_floor`.
- Experts keep the checkpoint's quant per tensor (NVFP4 with a per-expert `global_scale`, MXFP8) inside an affine
  pack. mlx-c's `gather_qmm` takes no global scale: the bank's output is multiplied by `global_scale / (6 * 448)` in
  f32 (`moe.zig`). `gate_up_proj` rows alternate gate and up.
- Images: no resize (the processor pads to whole 40-pixel blocks at -1/255; an image over the engine's pixel budget
  is shrunk first), one row per block, spliced AFTER the embedding norm. The host renders no-think (`reasoning_effort`
  none = `Thinking effort level: 0.0`) unless the client asks; the reference's default is 0.9.
- Serial (`batched_rows` refused). A verify forward leaves `.conv_input`, from which PLD's and the drafter's partial
  accepts roll the conv windows back.
- Drafter: the checkpoint's DSpark (`mtp.safetensors` beside the pack; `mtp_config` in the trunk's `config.json`) is
  the engine's block drafter in its tap-mix variant (`spec/dflash`, `DflashConfig.tap_mix`). Its taps are the residual
  ENTERING visits 5, 23 and 41 (captured as the outputs of 4, 22, 40) and the final-normed output, published at
  capture id = the visit count. The mlx-community 4-bit pack ships no drafter file: copy the original repo's.
- A verify row (PLD's or the drafter's) and a decode row round differently (a verify chunk under 9 rows takes the
  fp32 op chain, a decode row MLX's fused attention), so a tie between two tokens can fall the other way: drafted
  text is not serial's byte for byte, and `tests/test_inkling2.sh` holds it to "the list continues", not a hash. Oracle: `tests/dump_inkling2_drafter_fixtures.py` (oMLX's drafter modules, vLLM's context windows).
- Declined by name: a variable loop, shared experts, a core off the block boundaries, `use_value_norm_no_gamma` or
  `attn_res_normalize_values` off, q/o biases, an oMLX-layout pack (experts `w13`/`w2`, `Inkling-2-Mini-oQ4e-mtp`).
- A prompt chunk's experts: a per-expert tiled qmm lost to MLX's padded sorted gather on non-NAX.
- Oracle: mlx-lm PR #2086 on `mlx-community/Inkling-2-Mini-4bit` and its bf16 (`mlx_lm.convert -d`: dense banks)
  (`tests/dump_inkling2_fixtures.py`, `tests/test_inkling2.sh`). A prompt's bar is the reference's own kernel-vs-op-chain spread (`self_gap`, floor
  0.75): a rounding-equivalent change moves a long prompt's worst gap by 0.1-0.3, so no fixed bar tells it from a bug.
