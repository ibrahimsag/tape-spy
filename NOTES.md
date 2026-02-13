# microgpt C port — project notes

## What exists

- `arena.h` — stb-style arena allocator (virtual memory backed, reserve/commit, scratch pool). From Magicalbat/videos, added `__APPLE__` support and implementation guard.
- `autograd.h` — stb-style scalar autograd tape. Nodes are 32 bytes (padded for arena alignment). Creation order = topological order, so backward is a reverse linear scan with zero extra allocation. Tested: z=x*y+x gives correct gradients.
- `ui.h` — minimal immediate-mode UI on SDL_Renderer + stb_truetype. Has text, buttons, rect, line, heatmap cell. Font atlas baked with stb_truetype.
- `main.c` — MNIST IDX loader, SDL3 window with grid viewer, right panel with buttons (prev/next/train), screen state machine (VIEW/TRAINING/RESULTS).
- `stb_truetype.h` — copied from ../li
- `SourceCodePro-Regular.ttf` — copied from ../li
- MNIST data in `data/` subdir (IDX format, gunzipped from Google Cloud mirror)
- `build.sh` — `cc -O2 -o mnist_view main.c $(pkg-config --cflags --libs sdl3)`

## Text rendering / HiDPI issue (unresolved)

Text looks rough on macOS Retina. Current approach:
- `SDL_SetRenderLogicalPresentation(renderer, win_w, win_h, SDL_LOGICAL_PRESENTATION_LETTERBOX)` for logical coordinates
- Font baked at `font_size * dpi_scale * 2` (64px on 2x Retina for 16pt text)
- Quads scaled by `font_size / bake_size`
- Added `SDL_SetTextureScaleMode(ctx->font_tex, SDL_SCALEMODE_LINEAR)`
- Atlas is 1024x1024

If still rough: try dropping `SDL_SetRenderLogicalPresentation` entirely, work in pixel coordinates, scale all layout by dpi_scale manually. The logical presentation may be doing integer scaling that pixelates. The ../li project avoids this by using SDL_GPU with SDF shaders.

## Architecture — autograd tape

- `Node { data, grad, children[2], local_grads[2], arity, _pad }` — 32 bytes
- `Tape { arena, base, count, perm_count }` — owns dedicated arena
- Parameter nodes at front (indices 0..perm_count-1), survive tape_reset
- Computation nodes after, freed on tape_reset
- Backward: reverse linear scan from loss_idx to 0, accumulate `local_grad * grad` into children. No toposort needed.
- Ops: add, sub, mul, neg, pow, log, exp, relu, div (= mul + pow(-1))
- `_pad` field (u32) could store a network position tag for debugging

## Next steps

1. **Fix text rendering** — either fix the logical presentation approach or switch to pixel coordinates
2. **MNIST MLP training loop** — define 784→128→10 MLP using autograd tape, forward/backward/SGD, verify loss goes down
3. **Training on separate thread** — `TrainStatus` struct with atomics (step, epoch, loss, accuracy, running, request), main thread reads for display, train thread writes
4. **Wire training into UI** — Train button kicks off thread, training screen shows live loss/accuracy
5. **GPT** — port microgpt.py using same autograd. Adds: embeddings, rmsnorm, multi-head attention (Q/K/V projections, scaled dot-product, causal mask), residual connections. Attention is O(n²) in sequence length (actually n(n+1)/2 due to causal mask). KV cache grows per position on tape.

## Key design decisions

- Scalar autograd (like micrograd), not matrix-level (like the videos-main MNIST code or PyTorch). More nodes but simpler, uniform, educational.
- Arena-backed tape eliminates toposort — Python micrograd needs DFS because it has no creation-order list; arena IS that list.
- MLP is boring but necessary — same linear→relu→linear structure appears as the MLP block in each transformer layer. Not a detour.
- Character-level tokenization for GPT (no BPE). Fine for names dataset (27 chars). BPE is just compression, bolted on when sequences get long.

## Reference code

- `microgpt.py` — Karpathy's pure-Python GPT, the target to port
- `../root_basic/code/base/base_arena.*` — Ryan Fleury's arena (has chaining, ASAN annotations, arena naming, injected platform layer). More sophisticated than what we use but same core pattern.
- `../li/draw_list.*` — GPU-accelerated draw list with SDF text. Too heavy to bring over but good reference for future rendering upgrade.
- `videos-main/machine-learning/` — C MNIST with matrix-level autograd, static graph, arena + scratch. Hybrid approach (graph walk automated, backward per-op manual).
