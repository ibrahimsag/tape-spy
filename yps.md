# Tape spy plot — interactive viewer design

## Current state
- `dump_tape_spy` writes a fixed-resolution PPM (OKLCH hue-mapped by edge density)
- Transposed: x=child (to), y=node (from). Sorted dimension (from) is on y-axis.
- Triggered by D key during training (training thread does the dump after forward+backward)
- 4096x4096, ~48MB file. No interactive zoom yet.

## Axes (transposed)
- **x-axis (columns)**: child index (`to`) — the node being referenced
- **y-axis (rows)**: node index (`from`) — the node doing the referencing
- Each row has ≤2 entries (each node has ≤2 children)

## Storage: CSR per level

Each level is a CSR (compressed sparse row) matrix:
- `row_ptr[num_rows+1]`: offset into col_idx/counts for each row
- `col_idx[]`: column indices
- `counts[]`: accumulated edge counts

Why CSR:
- O(1) to start of any row's entries via `row_ptr[r]`
- At level 0, each row has exactly `arity` entries (0, 1, or 2) — trivial to build
- At coarser levels rows grow (≤4, ≤8...) so `row_ptr` earns its keep
- Merge operation walks level k's CSR left to right, builds level k+1's `row_ptr` incrementally

### Level 0 (finest)
- Walk tape nodes in order, emit `(from, children[c])` — already sorted by row
- Each row has ≤2 entries, sorted by col (children emitted in order)
- No separate edge buffer needed — built directly from tape snapshot

### Building coarser levels
Each level halves both dimensions: `(row, col) → (row>>1, col>>1)`.

One linear pass per level:
- Walk level k's CSR, emit entries with `(row>>1, col>>1)`, sum counts for duplicates
- Level 0→1: two rows of ≤2 entries merge to ≤4; higher levels can grow (≤2^(k+1) before col collapsing)
- Build level k+1's row_ptr incrementally as each merged row completes
- Output remains sorted — both shifts preserve monotonicity

Entry count per level: bounded by min(2*edges, (node_count/2^k)^2).
At coarse levels the quadratic term dominates and entries collapse fast.
Number of levels: ceil(log2(node_count)) — ~21 for 1.18M nodes, coarse ones are tiny.

## SpyCtx struct

```c
typedef struct {
    u32 row_count;
    u32 nnz;          // number of non-zero entries
    u32 *row_ptr;     // [row_count + 1]
    u32 *col_idx;     // [nnz]
    u32 *counts;      // [nnz]
} SpyLevel;

typedef struct {
    mem_arena *arena;      // owns all level arrays, arena_clear to rebuild
    SpyLevel *levels;
    u32 num_levels;
    u32 node_count;        // original tape node count
} SpyCtx;
```

Lives on TrainCtx. State machine via atomic:
- `idle` → main thread sets `request`
- `request` → training thread sees it after next forward+backward
- `working` → training thread builds all levels into arena
- `ready` → main thread reads CSR levels for rendering

Next request: `arena_clear` and rebuild from scratch.

## Rendering (main thread)

Main thread owns the renderer, does all SDL calls.
When state is `ready`, CSR data is immutable — no sync needed beyond the atomic flag.

1. Select level (manual up/down arrows initially, tie to zoom later)
2. Look up `row_ptr[visible_row_start]` upto `row_ptr[visible_row_end]`
3. Iterate entries in range into screen-sized u16 count buffer
4. OKLCH color pass
5. `SDL_UpdateTexture`
6. Cache texture — only re-render on level/viewport change

Display current level index + row/col counts for orientation.

## Performance budget
- ~10ns per edge (coord math, bounds check, u16 increment)
- Target <8ms per render → ~800K edges per viewport update
- At 800px wide viewport: ~500 nodes per pixel column is the crossover
- Pyramid build: linear passes, one-shot on snapshot, fine to stall training

## Scale estimates
| Architecture     | Nodes  | Edges  | CSR level 0 | All levels (est.) |
|-----------------|--------|--------|-------------|-------------------|
| MLP 784→128→10  | 306K   | 407K   | ~5MB        | ~8MB              |
| CNN 3-conv      | 1.18M  | 1.4M   | ~17MB       | ~28MB             |
| 10M nodes       | 10M    | ~20M   | ~240MB      | ~400MB            |

All level arrays live in one arena. arena_clear tosses everything, no individual frees.
CSR built directly from tape snapshot — no separate edge buffer.

## Controls
- D: request spy snapshot (during training)
- Up/Down: step through pyramid levels
- Mouse wheel: zoom (later, tied to level selection)
- Click+drag: pan
- ESC: back to training screen
- Diagonal reference line (dim gray where row == col)
