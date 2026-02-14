# Tape spy plot — interactive viewer design

## Current state
- `dump_tape_spy` writes a fixed-resolution PPM (OKLCH hue-mapped by edge density)
- Transposed: x=child (to), y=node (from). Sorted dimension (from) is on y-axis.
- Triggered by D key during training (training thread does the dump after forward+backward)
- 4096x4096, ~48MB file. No interactive zoom yet.

## Axes (transposed)
- **x-axis (columns)**: child index (`to`) — the node being referenced
- **y-axis (rows)**: node index (`from`) — the node doing the referencing
- Binary search on `from` (y) gives visible row range directly
- Column dimension (`to`) is where the pyramid helps

## Sparse pyramid

### Representation
Each level stores a sorted list of `(col, row, count)` tuples — only non-empty cells.
Memory is proportional to edge count, not pixel count. No N^2 dense grids.

### Level 0 (finest)
- One entry per edge: `(to, from, 1)`, sorted by `(to, from)`
- Deduplicate: merge entries with same `(to, from)`, sum counts
- Each node has at most 2 children — can emit them in sorted order

### Building coarser levels
Each level halves both dimensions: `(row, col) → (row>>1, col>>1)`.

One linear pass per level:
- Walk sorted entries, apply `(row>>1, col>>1)`, sum counts for matching cells
- Each row has ≤2 entries, so merging two adjacent rows gives ≤4 entries
- `col>>1` may collapse some of those (children in adjacent columns merge)
- Deduplicate in place — merged rows are tiny, no buffering needed
- Output remains sorted by `(col, row)` — both shifts preserve monotonicity

Entry count per level: bounded by min(2*edges, (node_count/2^k)^2).
At coarse levels the quadratic term dominates and entries collapse fast.
Total work across all levels: linear passes, purely sequential access.

### Rendering a viewport
1. Pick the level where column width ≈ pixel width
2. Binary search for visible row range (sorted by row within each column)
3. Iterate tuples in range, write into screen-sized u16 count buffer
4. OKLCH color pass, upload to SDL texture
5. Cache texture — only re-render on viewport change

At coarse levels the tuple list is short → fast even fully zoomed out.
At fine levels binary search on rows skips most data → fast when zoomed in.

### Crossover to direct iteration
~500 nodes per pixel column is the threshold for direct edge iteration (~8ms budget
at 800px wide viewport, ~10ns per edge). Below that, skip the pyramid and iterate
raw edges for the visible y-range.

## Performance budget
- ~10ns per edge (coord math, bounds check, u16 increment)
- Target <8ms per render → ~800K edges per viewport update
- Pyramid build: linear passes, ~2x edge count total, one-shot on snapshot

## Scale estimates
| Architecture     | Nodes  | Edges  | Edge buffer | Pyramid overhead |
|-----------------|--------|--------|-------------|-----------------|
| MLP 784→128→10  | 306K   | 407K   | 3.3MB       | ~5MB            |
| CNN 3-conv      | 1.18M  | 1.4M   | 11MB        | ~16MB           |
| 10M nodes       | 10M    | ~20M   | 160MB       | ~240MB          |

Edge snapshot memory ≈ half the tape size. No hard wall — degrades gracefully.
Fine to pause training for the snapshot.

## Controls
- Mouse wheel: zoom (centered on cursor)
- Click+drag: pan
- ESC: back to training screen
- Diagonal reference line (dim gray where to == from)
