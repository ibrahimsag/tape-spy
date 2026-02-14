# microgpt C port — project notes

## What exists

- `arena.h` — stb-style arena allocator (virtual memory backed, reserve/commit, scratch pool). From Magicalbat/videos, added `__APPLE__` support and implementation guard.
- `autograd.h` — stb-style scalar autograd tape. Nodes are 32 bytes (padded for arena alignment). Creation order = topological order, so backward is a reverse linear scan with zero extra allocation. `tape_reset` frees computation nodes; `tape_zero_grad` zeros param gradients (split for batching).
- `ui.h` — minimal immediate-mode UI on SDL_Renderer + stb_truetype. Has text, buttons, rect, line, heatmap cell. Font atlas baked with stb_truetype.
- `main.c` — MNIST CNN trainer with SDL3 UI. IDX loader, grid viewer, training thread with live stats + test accuracy graph.
- `stb_truetype.h` — copied from ../li
- `SourceCodePro-Regular.ttf` — copied from ../li
- MNIST data in `data/` subdir (IDX format, gunzipped from Google Cloud mirror)
- `build.sh` — `cc -O2 -o mnist_view main.c $(pkg-config --cflags --libs sdl3)`

## Text rendering — resolved

Fixed by adding `SDL_WINDOW_HIGH_PIXEL_DENSITY` to `SDL_CreateWindow`. Without it, SDL3 creates a 1x framebuffer even on Retina — the supersampled font atlas was being rendered into a low-res buffer that the OS then stretched. With the flag, the framebuffer matches native pixel density and the 2x supersampled atlas actually has 2x physical pixels to work with.

## Architecture — autograd tape

- `Node { data, grad, children[2], local_grads[2], arity, _pad }` — 32 bytes
- `Tape { arena, base, count, perm_count }` — owns dedicated arena
- Parameter nodes at front (indices 0..perm_count-1), survive tape_reset
- Computation nodes after, freed on tape_reset
- Backward: reverse linear scan from loss_idx to 0, accumulate `local_grad * grad` into children. No toposort needed.
- Ops: add, sub, mul, neg, pow, log, exp, relu, div (= mul + pow(-1))

## Adam optimizer — zero extra allocation

Parameter nodes have arity=0, so `local_grads[0]` and `local_grads[1]` are never read during backward (inner loop skips when arity=0). We store Adam's first moment (m) and second moment (v) in these fields. `tape_reset` doesn't touch local_grads, so Adam state survives across steps. Bias correction via running `b1_t *= 0.9` and `b2_t *= 0.999` each step (avoids powf).

## CNN architecture (current)

4 conv layers + FC:
- Conv1: 1→16, 5×5, stride 2  (28×28 → 12×12)
- Conv2: 16→32, 3×3, stride 1  (12×12 → 10×10)
- Conv3: 32→64, 3×3, stride 2  (10×10 → 4×4)
- Conv4: 64→128, 3×3, stride 1  (4×4 → 2×2)
- FC: 512→10

Conv weights shared across spatial positions — same param index referenced by val_mul at every slide position. Backward accumulates gradients from all positions into shared weights naturally.

## Training

- Batched (B=32): accumulate gradients across batch, average, then Adam step
- `tape_reset` frees computation nodes between samples (keeps param grads)
- `tape_zero_grad` zeros param grads after each Adam update
- Threaded: pthread runs training loop, main thread reads atomics for UI display
- Test accuracy evaluated every 10K steps on full 10K test set
- Test accuracy history in ring buffer (32 entries), displayed as line graph

## Results

- MLP 784→128→10: ~97% test accuracy ceiling, gets there in 1-2 epochs with Adam
- MLP 784→128→64→10: same ceiling, harder to train (vanishing gradients, needed lower lr)
- CNN 2-layer (16,32 filters): reached 98% test, got there faster than MLP with fewer params (~13K vs 110K)
- CNN 3-layer (16,32,64): similar ceiling, slightly faster initial convergence
- CNN 4-layer (16,32,64,128): starts faster, same ~98% ceiling. MNIST is too simple for deeper nets.
- Adam vs SGD: Adam reaches plateau much faster (2 epochs vs 4-5), but oscillates more with batch size 1. Batching (B=32) smooths it out.
- Overfitting demo: training on 10K images → 100% train acc, 95.7% test acc. Gap is the overfit. MNIST too clean for dramatic degradation.

## Key design decisions

- Scalar autograd (like micrograd), not matrix-level. More nodes but simpler, uniform, educational.
- Arena-backed tape eliminates toposort — Python micrograd needs DFS because it has no creation-order list; arena IS that list.
- Conv layers work naturally with scalar autograd — weight sharing is just multiple val_mul nodes referencing the same param index.
- Character-level tokenization for GPT (no BPE). Fine for names dataset (27 chars). BPE is just compression, bolted on when sequences get long.

## Next steps

1. **GPT** — port microgpt.py using same autograd. Adds: embeddings, rmsnorm, multi-head attention (Q/K/V projections, scaled dot-product, causal mask), residual connections. Attention is O(n²) in sequence length (actually n(n+1)/2 due to causal mask).

## Reference code

- `microgpt.py` — Karpathy's pure-Python GPT, the target to port
- `../root_basic/code/base/base_arena.*` — Ryan Fleury's arena (has chaining, ASAN annotations, arena naming, injected platform layer). More sophisticated than what we use but same core pattern.
- `../li/draw_list.*` — GPU-accelerated draw list with SDF text. Too heavy to bring over but good reference for future rendering upgrade.
- `videos-main/machine-learning/` — C MNIST with matrix-level autograd, static graph, arena + scratch. Hybrid approach (graph walk automated, backward per-op manual).
