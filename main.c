#define ARENA_IMPLEMENTATION
#include "arena.h"
#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>
#include <pthread.h>
#include <stdatomic.h>

#define UI_IMPLEMENTATION
#include "ui.h"

// --- IDX file loading ---

typedef struct {
    u32 count;
    u32 rows;
    u32 cols;
    u8 *pixels;  // count * rows * cols bytes, each 0-255
} idx_images;

typedef struct {
    u32 count;
    u8 *labels;  // count bytes, each 0-9
} idx_labels;

static u32 bswap32(u32 x) {
    return (x >> 24) | ((x >> 8) & 0xFF00) | ((x << 8) & 0xFF0000) | (x << 24);
}

static idx_images idx_load_images(mem_arena *arena, const char *path) {
    idx_images img = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "failed to open %s\n", path); return img; }

    u32 header[4];
    fread(header, 4, 4, f);
    img.count = bswap32(header[1]);
    img.rows  = bswap32(header[2]);
    img.cols  = bswap32(header[3]);

    u64 size = (u64)img.count * img.rows * img.cols;
    img.pixels = PUSH_ARRAY_NZ(arena, u8, size);
    fread(img.pixels, 1, size, f);
    fclose(f);
    return img;
}

static idx_labels idx_load_labels(mem_arena *arena, const char *path) {
    idx_labels lbl = {0};
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "failed to open %s\n", path); return lbl; }

    u32 header[2];
    fread(header, 4, 2, f);
    lbl.count = bswap32(header[1]);

    lbl.labels = PUSH_ARRAY_NZ(arena, u8, lbl.count);
    fread(lbl.labels, 1, lbl.count, f);
    fclose(f);
    return lbl;
}

// --- blit MNIST grid into pixel buffer ---

static void blit_grid(u32 *pixel_buf, int tex_w, int tex_h,
                      idx_images *images, u32 offset,
                      int grid_cols, int grid_rows) {
    int img_w = images->cols;
    int img_h = images->rows;
    memset(pixel_buf, 0, sizeof(u32) * tex_w * tex_h);
    for (int gy = 0; gy < grid_rows; gy++) {
        for (int gx = 0; gx < grid_cols; gx++) {
            u32 idx = offset + gy * grid_cols + gx;
            if (idx >= images->count) return;
            u8 *src = images->pixels + (u64)idx * img_w * img_h;
            for (int y = 0; y < img_h; y++) {
                for (int x = 0; x < img_w; x++) {
                    u8 v = src[y * img_w + x];
                    int px = gx * img_w + x;
                    int py = gy * img_h + y;
                    pixel_buf[py * tex_w + px] = 0xFF000000 | (v << 16) | (v << 8) | v;
                }
            }
        }
    }
}

// --- PRNG (LCG) ---

static u64 _rng_state = 0x12345678DEADBEEFULL;

static f32 rng_f32(void) {
    _rng_state = _rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (f32)((_rng_state >> 33) & 0x7FFFFF) / (f32)(1 << 23);
}

static f32 rng_normal(void) {
    f32 s = 0;
    for (int i = 0; i < 12; i++) s += rng_f32();
    return s - 6.0f;
}

// --- CNN (conv 5x5 -> conv 3x3 -> FC) ---

#define MLP_OUT 10

typedef struct {
    u32 start;
    u32 in_dim, out_dim;
} Layer;

typedef struct {
    u32 start;
    u32 in_c, out_c;
    u32 kh, kw;
    u32 stride;
} ConvLayer;

typedef struct {
    Tape tape;
    ConvLayer c1, c2, c3;
    Layer fc;
    u32 logits[MLP_OUT];
} MLP;

static MLP mlp_create(void) {
    MLP m;
    m.tape = tape_create(MiB(64));

    // Conv1: 1->16, 5x5, stride 2  (28x28 -> 12x12)
    m.c1 = (ConvLayer){ .start = m.tape.count, .in_c = 1, .out_c = 16, .kh = 5, .kw = 5, .stride = 2 };
    f32 s1 = sqrtf(2.0f / (5 * 5));
    for (u32 i = 0; i < 16 * 5 * 5; i++) tape_param(&m.tape, rng_normal() * s1);
    for (u32 i = 0; i < 16; i++) tape_param(&m.tape, 0.0f);

    // Conv2: 16->32, 3x3, stride 2  (12x12 -> 5x5)
    m.c2 = (ConvLayer){ .start = m.tape.count, .in_c = 16, .out_c = 32, .kh = 3, .kw = 3, .stride = 2 };
    f32 s2 = sqrtf(2.0f / (3 * 3 * 16));
    for (u32 i = 0; i < 32 * 3 * 3 * 16; i++) tape_param(&m.tape, rng_normal() * s2);
    for (u32 i = 0; i < 32; i++) tape_param(&m.tape, 0.0f);

    // Conv3: 32->64, 3x3, stride 1  (5x5 -> 3x3)
    m.c3 = (ConvLayer){ .start = m.tape.count, .in_c = 32, .out_c = 64, .kh = 3, .kw = 3, .stride = 1 };
    f32 s3 = sqrtf(2.0f / (3 * 3 * 32));
    for (u32 i = 0; i < 64 * 3 * 3 * 32; i++) tape_param(&m.tape, rng_normal() * s3);
    for (u32 i = 0; i < 64; i++) tape_param(&m.tape, 0.0f);

    // FC: 3*3*64=576 -> 10
    m.fc = (Layer){ .start = m.tape.count, .in_dim = 576, .out_dim = MLP_OUT };
    f32 s4 = sqrtf(2.0f / 576);
    for (u32 i = 0; i < MLP_OUT * 576; i++) tape_param(&m.tape, rng_normal() * s4);
    for (u32 i = 0; i < MLP_OUT; i++) tape_param(&m.tape, 0.0f);

    return m;
}

static void linear_fwd(Tape *t, Layer *l, u32 *in, u32 *out) {
    for (u32 i = 0; i < l->out_dim; i++) {
        u32 w = l->start + i * l->in_dim;
        u32 acc = val_mul(t, w, in[0]);
        for (u32 j = 1; j < l->in_dim; j++)
            acc = val_add(t, acc, val_mul(t, w + j, in[j]));
        out[i] = val_add(t, acc, l->start + l->out_dim * l->in_dim + i);
    }
}

// Conv2D forward: in[h][w][c] -> out[h][w][f]
static void conv_fwd(Tape *t, ConvLayer *l, u32 *in, u32 in_h, u32 in_w, u32 *out) {
    u32 out_h = (in_h - l->kh) / l->stride + 1;
    u32 out_w = (in_w - l->kw) / l->stride + 1;
    u32 ksz = l->kh * l->kw * l->in_c;

    for (u32 f = 0; f < l->out_c; f++) {
        u32 w_base = l->start + f * ksz;
        u32 bias_idx = l->start + l->out_c * ksz + f;
        for (u32 oy = 0; oy < out_h; oy++) {
            u32 iy0 = oy * l->stride;
            for (u32 ox = 0; ox < out_w; ox++) {
                u32 ix0 = ox * l->stride;
                u32 acc = 0, w = 0;
                for (u32 ky = 0; ky < l->kh; ky++) {
                    for (u32 kx = 0; kx < l->kw; kx++) {
                        for (u32 c = 0; c < l->in_c; c++) {
                            u32 in_idx = (iy0+ky) * in_w * l->in_c + (ix0+kx) * l->in_c + c;
                            u32 prod = val_mul(t, w_base + w, in[in_idx]);
                            acc = (w == 0) ? prod : val_add(t, acc, prod);
                            w++;
                        }
                    }
                }
                out[oy * out_w * l->out_c + ox * l->out_c + f] = val_add(t, acc, bias_idx);
            }
        }
    }
}

// Forward pass: returns loss node index, fills m->logits
static u32 mlp_forward(MLP *m, u8 *pixels, u8 label) {
    Tape *t = &m->tape;

    // Input leaves: 28x28x1
    u32 inp[28 * 28];
    for (u32 i = 0; i < 28 * 28; i++)
        inp[i] = tape_leaf(t, (f32)pixels[i] / 255.0f);

    // Conv1 + ReLU: 28x28x1 -> 12x12x16
    u32 c1[12 * 12 * 16];
    conv_fwd(t, &m->c1, inp, 28, 28, c1);
    for (u32 i = 0; i < 12 * 12 * 16; i++) c1[i] = val_relu(t, c1[i]);

    // Conv2 + ReLU: 12x12x16 -> 5x5x32
    u32 c2[5 * 5 * 32];
    conv_fwd(t, &m->c2, c1, 12, 12, c2);
    for (u32 i = 0; i < 5 * 5 * 32; i++) c2[i] = val_relu(t, c2[i]);

    // Conv3 + ReLU: 5x5x32 -> 3x3x64
    u32 c3[3 * 3 * 64];
    conv_fwd(t, &m->c3, c2, 5, 5, c3);
    for (u32 i = 0; i < 3 * 3 * 64; i++) c3[i] = val_relu(t, c3[i]);

    // FC: 576 -> 10
    linear_fwd(t, &m->fc, c3, m->logits);

    // Cross-entropy via log-softmax (numerically stable)
    f32 max_v = tape_data(t, m->logits[0]);
    for (u32 i = 1; i < MLP_OUT; i++) {
        f32 v = tape_data(t, m->logits[i]);
        if (v > max_v) max_v = v;
    }
    u32 mx = tape_leaf(t, max_v);

    u32 se = val_exp(t, val_sub(t, m->logits[0], mx));
    for (u32 i = 1; i < MLP_OUT; i++)
        se = val_add(t, se, val_exp(t, val_sub(t, m->logits[i], mx)));

    return val_sub(t, val_log(t, se), val_sub(t, m->logits[label], mx));
}

typedef struct { f32 b1_t, b2_t; } AdamState;

static void adam_step(Tape *t, f32 lr, AdamState *st) {
    f32 b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    st->b1_t *= b1;
    st->b2_t *= b2;
    f32 bc1 = 1.0f - st->b1_t;
    f32 bc2 = 1.0f - st->b2_t;
    for (u32 i = 0; i < t->perm_count; i++) {
        Node *n = &t->base[i];
        f32 g = n->grad;
        n->local_grads[0] = b1 * n->local_grads[0] + (1 - b1) * g;
        n->local_grads[1] = b2 * n->local_grads[1] + (1 - b2) * g * g;
        f32 m_hat = n->local_grads[0] / bc1;
        f32 v_hat = n->local_grads[1] / bc2;
        n->data -= lr * m_hat / (sqrtf(v_hat) + eps);
    }
}

static int mlp_predict(MLP *m) {
    int pred = 0;
    f32 best = tape_data(&m->tape, m->logits[0]);
    for (u32 i = 1; i < MLP_OUT; i++) {
        f32 v = tape_data(&m->tape, m->logits[i]);
        if (v > best) { best = v; pred = (int)i; }
    }
    return pred;
}

// --- OKLCH color ---

static float _srgb_gamma(float c) {
    if (c <= 0.0f) return 0.0f;
    if (c >= 1.0f) return 1.0f;
    return c <= 0.0031308f ? c * 12.92f : 1.055f * powf(c, 1.0f/2.4f) - 0.055f;
}

static void _oklch_to_rgb(float L, float C, float h_deg, u8 *ro, u8 *go, u8 *bo) {
    float h = h_deg * (3.14159265f / 180.0f);
    float a = C * cosf(h), b = C * sinf(h);

    float l_ = L + 0.3963377774f * a + 0.2158037573f * b;
    float m_ = L - 0.1055613458f * a - 0.0638541728f * b;
    float s_ = L - 0.0894841775f * a - 1.2914855480f * b;
    l_ = l_ * l_ * l_;
    m_ = m_ * m_ * m_;
    s_ = s_ * s_ * s_;

    float rf = +4.0767416621f * l_ - 3.3077115913f * m_ + 0.2309699292f * s_;
    float gf = -1.2684380046f * l_ + 2.6097574011f * m_ - 0.3413193965f * s_;
    float bf = -0.0041960863f * l_ - 0.7034186147f * m_ + 1.7076147010f * s_;

    *ro = (u8)(_srgb_gamma(rf) * 255.0f + 0.5f);
    *go = (u8)(_srgb_gamma(gf) * 255.0f + 0.5f);
    *bo = (u8)(_srgb_gamma(bf) * 255.0f + 0.5f);
}

// --- spy plot CSR pyramid ---

#define SPY_IDLE    0
#define SPY_REQUEST 1
#define SPY_WORKING 2
#define SPY_READY   3

typedef struct {
    u32 row_count;
    u32 col_count;
    u32 nnz;
    u32 *row_ptr;   // [row_count + 1]
    u32 *col_idx;   // [nnz]
    u32 *counts;    // [nnz]
} SpyLevel;

typedef struct {
    mem_arena *arena;
    SpyLevel *levels;
    u32 num_levels;
    u32 node_count;
    _Atomic int state;
} SpyCtx;

// Build level 0 CSR from tape snapshot
static void spy_build_level0(SpyCtx *spy, Tape *t) {
    u32 N = t->count;
    spy->node_count = N;

    SpyLevel *lv = &spy->levels[0];
    lv->row_count = N;
    lv->col_count = N;
    lv->row_ptr = PUSH_ARRAY(spy->arena, u32, N + 1);

    // first pass: count entries per row
    for (u32 i = 0; i <= N; i++) lv->row_ptr[i] = 0;
    u32 total = 0;
    for (u32 i = 0; i < N; i++) {
        lv->row_ptr[i] = total;
        total += t->base[i].arity;
    }
    lv->row_ptr[N] = total;
    lv->nnz = total;

    lv->col_idx = PUSH_ARRAY(spy->arena, u32, total);
    lv->counts  = PUSH_ARRAY(spy->arena, u32, total);

    // fill entries (children sorted: smaller index first)
    u32 pos = 0;
    for (u32 i = 0; i < N; i++) {
        Node *n = &t->base[i];
        if (n->arity == 2 && n->children[0] > n->children[1]) {
            lv->col_idx[pos] = n->children[1]; lv->counts[pos++] = 1;
            lv->col_idx[pos] = n->children[0]; lv->counts[pos++] = 1;
        } else {
            for (u32 c = 0; c < n->arity; c++) {
                lv->col_idx[pos] = n->children[c];
                lv->counts[pos++] = 1;
            }
        }
    }
}

// Build level 0 from random data for testing
static void spy_build_random(SpyCtx *spy, u32 N) {
    spy->node_count = N;

    SpyLevel *lv = &spy->levels[0];
    lv->row_count = N;
    lv->col_count = N;

    // each row gets 0-2 entries
    u32 total = 0;
    lv->row_ptr = PUSH_ARRAY(spy->arena, u32, N + 1);
    for (u32 i = 0; i < N; i++) {
        lv->row_ptr[i] = total;
        u32 arity = (i == 0) ? 0 : (rng_f32() < 0.1f ? 1 : 2);
        total += arity;
    }
    lv->row_ptr[N] = total;
    lv->nnz = total;

    lv->col_idx = PUSH_ARRAY(spy->arena, u32, total);
    lv->counts  = PUSH_ARRAY(spy->arena, u32, total);

    u32 pos = 0;
    for (u32 i = 0; i < N; i++) {
        u32 arity = lv->row_ptr[i + 1] - lv->row_ptr[i];
        u32 cols[2];
        for (u32 c = 0; c < arity; c++) {
            // bias children toward nearby indices (diagonal structure)
            u32 spread = i < 100 ? i : 100;
            cols[c] = (u32)((i > spread ? i - spread : 0) + (u32)(rng_f32() * spread * 2));
            if (cols[c] >= N) cols[c] = N - 1;
        }
        // sort
        if (arity == 2 && cols[0] > cols[1]) { u32 tmp = cols[0]; cols[0] = cols[1]; cols[1] = tmp; }
        for (u32 c = 0; c < arity; c++) {
            lv->col_idx[pos] = cols[c];
            lv->counts[pos++] = 1;
        }
    }
}

// Build coarser level k+1 from level k
static void spy_build_coarser(SpyCtx *spy, u32 k) {
    SpyLevel *src = &spy->levels[k];
    SpyLevel *dst = &spy->levels[k + 1];

    dst->row_count = (src->row_count + 1) / 2;
    dst->col_count = (src->col_count + 1) / 2;
    dst->row_ptr = PUSH_ARRAY(spy->arena, u32, dst->row_count + 1);

    // scratch arena for temp arrays (conflict with spy->arena so we get a different one)
    mem_arena *conflicts[] = { spy->arena };
    mem_arena_temp scratch = arena_scratch_get(conflicts, 1);

    u32 *out_cols   = PUSH_ARRAY(scratch.arena, u32, src->nnz);
    u32 *out_counts = PUSH_ARRAY(scratch.arena, u32, src->nnz);
    u32 out_pos = 0;

    // merge buffer sized for max possible per merged row
    u32 merge_cap = 0;
    for (u32 sr = 0; sr < src->row_count; sr++) {
        u32 n = src->row_ptr[sr + 1] - src->row_ptr[sr];
        if (n > merge_cap) merge_cap = n;
    }
    merge_cap = merge_cap * 2 + 1;
    u32 *merged_cols = PUSH_ARRAY(scratch.arena, u32, merge_cap);
    u32 *merged_cnts = PUSH_ARRAY(scratch.arena, u32, merge_cap);

    for (u32 dr = 0; dr < dst->row_count; dr++) {
        dst->row_ptr[dr] = out_pos;
        u32 sr0 = dr * 2;
        u32 sr1 = dr * 2 + 1;

        // collect entries from both source rows, applying col>>1
        u32 mc = 0;
        for (u32 sr = sr0; sr <= sr1 && sr < src->row_count; sr++) {
            u32 start = src->row_ptr[sr];
            u32 end   = src->row_ptr[sr + 1];
            for (u32 j = start; j < end; j++) {
                merged_cols[mc] = src->col_idx[j] >> 1;
                merged_cnts[mc] = src->counts[j];
                mc++;
            }
        }

        // sort by col (insertion sort — small per row)
        for (u32 i = 1; i < mc; i++) {
            u32 c = merged_cols[i], v = merged_cnts[i];
            u32 j = i;
            while (j > 0 && merged_cols[j-1] > c) {
                merged_cols[j] = merged_cols[j-1];
                merged_cnts[j] = merged_cnts[j-1];
                j--;
            }
            merged_cols[j] = c;
            merged_cnts[j] = v;
        }

        // deduplicate into output
        for (u32 i = 0; i < mc; i++) {
            if (out_pos > dst->row_ptr[dr] && out_cols[out_pos - 1] == merged_cols[i]) {
                out_counts[out_pos - 1] += merged_cnts[i];
            } else {
                out_cols[out_pos] = merged_cols[i];
                out_counts[out_pos] = merged_cnts[i];
                out_pos++;
            }
        }
    }
    dst->row_ptr[dst->row_count] = out_pos;
    dst->nnz = out_pos;

    // copy exact-sized results into persistent arena
    dst->col_idx = PUSH_ARRAY(spy->arena, u32, out_pos);
    dst->counts  = PUSH_ARRAY(spy->arena, u32, out_pos);
    memcpy(dst->col_idx, out_cols, out_pos * sizeof(u32));
    memcpy(dst->counts,  out_counts, out_pos * sizeof(u32));

    arena_scratch_release(scratch);
}

// Build all levels
static void spy_build_pyramid(SpyCtx *spy) {
    u32 max_dim = spy->levels[0].row_count;
    spy->num_levels = 1;
    while (max_dim > 1 && spy->num_levels < 24) {
        spy_build_coarser(spy, spy->num_levels - 1);
        spy->num_levels++;
        max_dim = (max_dim + 1) / 2;
    }
    printf("spy: built %u levels, level 0: %u rows, %u nnz\n",
           spy->num_levels, spy->levels[0].row_count, spy->levels[0].nnz);
}

// Render a level into an RGBA pixel buffer (1024x1024)
#define SPY_CANVAS 1024

typedef struct {
    u32 tex_w, tex_h;       // texture dimensions (pixels)
    float quad_x, quad_y;   // quad position on canvas (pixels, may be negative)
    float quad_w, quad_h;   // quad size on canvas (pixels, may exceed canvas)
} SpyRenderResult;

static SpyRenderResult spy_render_level(SpyCtx *spy, u32 level_idx,
                                         float cam_x, float cam_y, float cam_span,
                                         u32 *rgba) {
    SpyRenderResult res = {0};
    if (level_idx >= spy->num_levels) return res;
    SpyLevel *lv = &spy->levels[level_idx];

    // visible range in level's row/col space
    float r0f = cam_y * lv->row_count;
    float r1f = (cam_y + cam_span) * lv->row_count;
    float c0f = cam_x * lv->col_count;
    float c1f = (cam_x + cam_span) * lv->col_count;

    // snap to whole cells (expand outward)
    u32 r0 = (u32)r0f;
    u32 r1 = (u32)r1f;
    if (r1f > (float)r1 || r1 == r0) r1++;  // ceil, at least 1
    if (r1 > lv->row_count) r1 = lv->row_count;

    u32 c0 = (u32)c0f;
    u32 c1 = (u32)c1f;
    if (c1f > (float)c1 || c1 == c0) c1++;
    if (c1 > lv->col_count) c1 = lv->col_count;

    u32 vis_rows = r1 - r0;
    u32 vis_cols = c1 - c0;

    // texture dimensions: 1 texel per cell (capped at canvas)
    u32 tw = vis_cols < SPY_CANVAS ? vis_cols : SPY_CANVAS;
    u32 th = vis_rows < SPY_CANVAS ? vis_rows : SPY_CANVAS;
    if (tw == 0) tw = 1;
    if (th == 0) th = 1;
    res.tex_w = tw;
    res.tex_h = th;

    // quad: snapped range mapped to canvas coordinates
    float snap_x0 = (float)c0 / lv->col_count;
    float snap_y0 = (float)r0 / lv->row_count;
    float snap_x1 = (float)c1 / lv->col_count;
    float snap_y1 = (float)r1 / lv->row_count;

    res.quad_x = (snap_x0 - cam_x) / cam_span * SPY_CANVAS;
    res.quad_y = (snap_y0 - cam_y) / cam_span * SPY_CANVAS;
    res.quad_w = (snap_x1 - snap_x0) / cam_span * SPY_CANVAS;
    res.quad_h = (snap_y1 - snap_y0) / cam_span * SPY_CANVAS;

    // clear
    memset(rgba, 0, tw * th * sizeof(u32));

    // accumulate into u16 count buffer
    u16 *buf = calloc(tw * th, sizeof(u16));
    u16 max_count = 0;

    for (u32 r = r0; r < r1; r++) {
        int py = th > 1 ? (int)((u64)(r - r0) * (th - 1) / (vis_rows - 1)) : 0;
        u32 start = lv->row_ptr[r];
        u32 end   = lv->row_ptr[r + 1];
        for (u32 j = start; j < end; j++) {
            u32 col = lv->col_idx[j];
            if (col < c0 || col >= c1) continue;  // outside viewport
            int px = tw > 1 ? (int)((u64)(col - c0) * (tw - 1) / (vis_cols - 1)) : 0;
            int idx = py * tw + px;
            buf[idx] += (u16)(lv->counts[j] > 65535 - buf[idx] ? 65535 - buf[idx] : lv->counts[j]);
            if (buf[idx] > max_count) max_count = buf[idx];
        }
    }

    // OKLCH color pass
    for (u32 i = 0; i < tw * th; i++) {
        if (buf[i] == 0) {
            rgba[i] = 0xFF191919;
        } else {
            float frac = max_count > 1
                ? (float)(buf[i] - 1) / (float)(max_count - 1)
                : 0.0f;
            float hue = 260.0f - frac * 230.0f;
            u8 r, g, b;
            _oklch_to_rgb(0.75f, 0.15f, hue, &r, &g, &b);
            rgba[i] = 0xFF000000 | ((u32)b << 16) | ((u32)g << 8) | r;
        }
    }

    // diagonal reference: where row_index == col_index in node-space
    // in the visible window, diagonal passes through cells where (r0+ty) == (c0+tx)
    for (u32 ty = 0; ty < th; ty++) {
        // which row in level-space does this texel represent?
        u32 row = r0 + (vis_rows > 1 ? (u64)ty * (vis_rows - 1) / (th - 1) : 0);
        // diagonal col == row (since row_count == col_count)
        if (row < c0 || row >= c1) continue;
        int px = tw > 1 ? (int)((u64)(row - c0) * (tw - 1) / (vis_cols - 1)) : 0;
        int idx = ty * tw + px;
        if (buf[idx] == 0) {
            rgba[idx] = 0xFF282828;
        }
    }

    free(buf);
    return res;
}

static void dump_tape_spy(Tape *t, const char *path, int S) {
    u16 *counts = calloc(S * S, sizeof(u16));
    u16 max_count = 0;
    for (u32 i = 0; i < t->count; i++) {
        Node *n = &t->base[i];
        int y = (int)((u64)i * (S - 1) / t->count);          // row = from (node)
        for (u32 c = 0; c < n->arity; c++) {
            int x = (int)((u64)n->children[c] * (S - 1) / t->count); // col = to (child)
            int idx = y * S + x;
            counts[idx]++;
            if (counts[idx] > max_count) max_count = counts[idx];
        }
    }

    u8 *rgb = malloc(S * S * 3);
    for (int i = 0; i < S * S; i++) {
        if (counts[i] == 0) {
            rgb[i*3] = rgb[i*3+1] = rgb[i*3+2] = 0;
        } else {
            float frac = max_count > 1
                ? (float)(counts[i] - 1) / (float)(max_count - 1)
                : 0.0f;
            // hue: 260 (blue) -> 30 (orange) as density increases
            float hue = 260.0f - frac * 230.0f;
            _oklch_to_rgb(0.75f, 0.15f, hue, &rgb[i*3], &rgb[i*3+1], &rgb[i*3+2]);
        }
    }

    // mark the diagonal (y=x reference line)
    for (int i = 0; i < S; i++) {
        int idx = i * S + i;
        if (counts[idx] == 0) {
            rgb[idx*3] = 40; rgb[idx*3+1] = 40; rgb[idx*3+2] = 40;
        }
    }

    FILE *f = fopen(path, "wb");
    fprintf(f, "P6\n%d %d\n255\n", S, S);
    fwrite(rgb, 1, S * S * 3, f);
    fclose(f);
    free(counts);
    free(rgb);
    printf("spy plot: %s (%ux%u, %u nodes, max density %u)\n", path, S, S, t->count, max_count);
}

// --- training thread ---

#define ACC_HIST 32

typedef struct {
    _Atomic int  step;
    _Atomic int  epoch;
    _Atomic int  correct;
    _Atomic float loss;
    _Atomic float test_acc;
    _Atomic int  hist_count;
    float hist[ACC_HIST];       // ring buffer, written by train thread
    _Atomic bool running;
    _Atomic bool stop;
    _Atomic bool dump_requested;
    _Atomic int spy_state;  // 0=idle, 1=requested, 2=ready
    SpyCtx *spy;

    MLP *mlp;
    idx_images *images;
    idx_labels *labels;
    idx_images *test_images;
    idx_labels *test_labels;
    float lr;
} TrainCtx;

static void *train_thread_fn(void *arg) {
    TrainCtx *ctx = arg;
    atomic_store(&ctx->running, true);
    AdamState adam = { 1.0f, 1.0f };

    for (int epoch = 0; !atomic_load(&ctx->stop); epoch++) {
        atomic_store(&ctx->epoch, epoch);
        atomic_store(&ctx->step, 0);
        atomic_store(&ctx->correct, 0);
        Tape *tape = &ctx->mlp->tape;
        int B = 32;
        tape_zero_grad(tape);

        for (u32 i = 0; i < ctx->images->count && !atomic_load(&ctx->stop); i++) {
            u8 *px = ctx->images->pixels + (u64)i * 784;
            u8 label = ctx->labels->labels[i];

            tape_reset(tape);
            u32 loss_idx = mlp_forward(ctx->mlp, px, label);
            tape_backward(tape, loss_idx);

            if (atomic_load(&ctx->dump_requested)) {
                dump_tape_spy(tape, "tape_spy.ppm", 4096);
                atomic_store(&ctx->dump_requested, false);
            }

            if (atomic_load(&ctx->spy_state) == 1) {
                arena_clear(ctx->spy->arena);
                spy_build_level0(ctx->spy, tape);
                spy_build_pyramid(ctx->spy);
                atomic_store(&ctx->spy_state, 2);
            }

            if (mlp_predict(ctx->mlp) == label)
                atomic_fetch_add(&ctx->correct, 1);

            if ((i + 1) % B == 0) {
                f32 scale = 1.0f / B;
                for (u32 j = 0; j < tape->perm_count; j++)
                    tape->base[j].grad *= scale;
                adam_step(tape, ctx->lr, &adam);
                tape_zero_grad(tape);
            }

            atomic_store(&ctx->step, (int)i + 1);
            atomic_store(&ctx->loss, tape_data(tape, loss_idx));

            // evaluate on test set every 10K steps
            if ((i + 1) % 10000 == 0) {
                int test_correct = 0;
                for (u32 j = 0; j < ctx->test_images->count && !atomic_load(&ctx->stop); j++) {
                    u8 *tpx = ctx->test_images->pixels + (u64)j * 784;
                    u8 tlbl = ctx->test_labels->labels[j];
                    tape_reset(&ctx->mlp->tape);
                    mlp_forward(ctx->mlp, tpx, tlbl);
                    if (mlp_predict(ctx->mlp) == tlbl) test_correct++;
                }
                if (!atomic_load(&ctx->stop)) {
                    float ta = 100.0f * test_correct / (int)ctx->test_images->count;
                    atomic_store(&ctx->test_acc, ta);
                    int hc = atomic_load(&ctx->hist_count);
                    ctx->hist[hc % ACC_HIST] = ta;
                    atomic_store(&ctx->hist_count, hc + 1);
                }
                tape_zero_grad(tape);
            }
        }
    }

    atomic_store(&ctx->running, false);
    return NULL;
}

// --- app state ---

typedef enum {
    SCREEN_VIEW,
    SCREEN_TRAINING,
    SCREEN_SPY,
} Screen;

#define GRID_COLS 20
#define GRID_ROWS 10
#define GRID_COUNT (GRID_COLS * GRID_ROWS)

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;

    // --- autograd sanity test ---
    {
        Tape tape = tape_create(MiB(1));
        u32 x = tape_param(&tape, 3.0f);
        u32 y = tape_param(&tape, 4.0f);
        u32 xy = val_mul(&tape, x, y);
        u32 z  = val_add(&tape, xy, x);
        tape_backward(&tape, z);
        printf("autograd test: z = x*y + x, x=3, y=4\n");
        printf("  z     = %.1f (expect 15.0)\n", tape_data(&tape, z));
        printf("  dz/dx = %.1f (expect 5.0)\n", tape_grad(&tape, x));
        printf("  dz/dy = %.1f (expect 3.0)\n", tape_grad(&tape, y));
        tape_destroy(&tape);
    }

    mem_arena *arena = arena_create(MiB(256), MiB(1));

    idx_images train_images = idx_load_images(arena, "data/train-images-idx3-ubyte");
    idx_labels train_labels = idx_load_labels(arena, "data/train-labels-idx1-ubyte");
    idx_images test_images  = idx_load_images(arena, "data/t10k-images-idx3-ubyte");
    idx_labels test_labels  = idx_load_labels(arena, "data/t10k-labels-idx1-ubyte");

    if (!train_images.pixels || !train_labels.labels ||
        !test_images.pixels  || !test_labels.labels) {
        fprintf(stderr, "failed to load MNIST data\n");
        return 1;
    }
    printf("loaded %u train, %u test (%ux%u)\n",
           train_images.count, test_images.count,
           train_images.rows, train_images.cols);

    MLP mlp = mlp_create();
    printf("MLP: %u params (%zu bytes)\n",
           mlp.tape.perm_count, (size_t)mlp.tape.perm_count * sizeof(Node));

    TrainCtx train_ctx = {0};
    pthread_t train_thread;
    bool train_thread_active = false;

    SpyCtx spy = {0};
    spy.arena = arena_create(MiB(256), MiB(1));
    SpyLevel spy_levels[24] = {0};
    spy.levels = spy_levels;
    u32 spy_level_idx = 0;
    bool spy_dirty = false;
    float cam_x = 0, cam_y = 0, cam_span = 1.0f;
    bool cam_dragging = false;
    float drag_start_x, drag_start_y;  // mouse pos at drag start
    float drag_cam_x, drag_cam_y;      // cam pos at drag start

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    int img_w = train_images.cols;
    int img_h = train_images.rows;
    int panel_w = 280;
    int grid_pixel_w = img_w * GRID_COLS;
    int grid_pixel_h = img_h * GRID_ROWS;
    int win_w = 1024 + panel_w;
    int win_h = 1024;

    SDL_Window *window = SDL_CreateWindow("MNIST", win_w, win_h, SDL_WINDOW_HIGH_PIXEL_DENSITY);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_SetRenderVSync(renderer, 1);
    SDL_SetRenderLogicalPresentation(renderer, win_w, win_h,
                                     SDL_LOGICAL_PRESENTATION_LETTERBOX);

    // UI
    UICtx ui;
    float dpi_scale = SDL_GetWindowDisplayScale(window);
    if (dpi_scale < 1.0f) dpi_scale = 1.0f;
    if (!ui_init(&ui, renderer, "SourceCodePro-Regular.ttf", 16.0f, dpi_scale)) {
        fprintf(stderr, "ui_init failed\n");
        return 1;
    }

    // MNIST grid texture
    SDL_Texture *grid_tex = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING, grid_pixel_w, grid_pixel_h
    );

    u32 *pixel_buf = PUSH_ARRAY(arena, u32, grid_pixel_w * grid_pixel_h);
    u32 offset = 0;
    bool grid_dirty = true;

    // spy texture (variable size, recreated per level)
    SDL_Texture *spy_tex = NULL;
    u32 spy_tex_w = 0, spy_tex_h = 0;
    u32 *spy_pixels = PUSH_ARRAY(arena, u32, SPY_CANVAS * SPY_CANVAS);
    Screen prev_screen = SCREEN_VIEW;
    Screen screen = SCREEN_VIEW;

    bool running = true;
    while (running) {
        ui_begin(&ui);

        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            ui_event(&ui, &event);
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
            if (event.type == SDL_EVENT_KEY_DOWN) {
                if (screen == SCREEN_SPY && event.key.key == SDLK_ESCAPE) {
                    screen = prev_screen;
                } else if (event.key.key == SDLK_ESCAPE || event.key.key == SDLK_Q) {
                    running = false;
                }
                if (event.key.key == SDLK_D && screen == SCREEN_TRAINING) {
                    atomic_store(&train_ctx.dump_requested, true);
                }
                if (screen == SCREEN_SPY) {
                    if (event.key.key == SDLK_HOME) {
                        cam_x = cam_y = 0; cam_span = 1.0f;
                        spy_dirty = true;
                    }
                }
            }
            // spy mouse events
            if (screen == SCREEN_SPY) {
                if (event.type == SDL_EVENT_MOUSE_WHEEL) {
                    float mx = ui.mouse_x, my = ui.mouse_y;
                    if (mx >= 0 && mx < SPY_CANVAS && my >= 0 && my < SPY_CANVAS) {
                        // zoom centered on mouse position
                        float frac_x = mx / SPY_CANVAS;
                        float frac_y = my / SPY_CANVAS;
                        float zoom = event.wheel.y > 0 ? 0.8f : 1.25f;
                        float new_span = cam_span * zoom;
                        if (new_span < 1.0f / (1 << 20)) new_span = 1.0f / (1 << 20);
                        if (new_span > 1.0f) new_span = 1.0f;
                        // adjust cam_x/cam_y so mouse points at same node-space position
                        float node_x = cam_x + frac_x * cam_span;
                        float node_y = cam_y + frac_y * cam_span;
                        cam_x = node_x - frac_x * new_span;
                        cam_y = node_y - frac_y * new_span;
                        cam_span = new_span;
                        // clamp
                        if (cam_x < 0) cam_x = 0;
                        if (cam_y < 0) cam_y = 0;
                        if (cam_x + cam_span > 1.0f) cam_x = 1.0f - cam_span;
                        if (cam_y + cam_span > 1.0f) cam_y = 1.0f - cam_span;
                        spy_dirty = true;
                    }
                }
                if (event.type == SDL_EVENT_MOUSE_BUTTON_DOWN && event.button.button == SDL_BUTTON_LEFT) {
                    if (ui.mouse_x >= 0 && ui.mouse_x < SPY_CANVAS) {
                        cam_dragging = true;
                        drag_start_x = ui.mouse_x;
                        drag_start_y = ui.mouse_y;
                        drag_cam_x = cam_x;
                        drag_cam_y = cam_y;
                    }
                }
                if (event.type == SDL_EVENT_MOUSE_BUTTON_UP && event.button.button == SDL_BUTTON_LEFT) {
                    cam_dragging = false;
                }
                if (event.type == SDL_EVENT_MOUSE_MOTION && cam_dragging) {
                    float dx = (ui.mouse_x - drag_start_x) / SPY_CANVAS * cam_span;
                    float dy = (ui.mouse_y - drag_start_y) / SPY_CANVAS * cam_span;
                    cam_x = drag_cam_x - dx;
                    cam_y = drag_cam_y - dy;
                    if (cam_x < 0) cam_x = 0;
                    if (cam_y < 0) cam_y = 0;
                    if (cam_x + cam_span > 1.0f) cam_x = 1.0f - cam_span;
                    if (cam_y + cam_span > 1.0f) cam_y = 1.0f - cam_span;
                    spy_dirty = true;
                }
            }
            if (event.type == SDL_EVENT_KEY_DOWN && screen == SCREEN_VIEW) {
                if (event.key.key == SDLK_RIGHT || event.key.key == SDLK_SPACE) {
                    if (offset + GRID_COUNT < train_images.count) {
                        offset += GRID_COUNT;
                        grid_dirty = true;
                    }
                }
                if (event.key.key == SDLK_LEFT) {
                    if (offset >= GRID_COUNT) offset -= GRID_COUNT;
                    else offset = 0;
                    grid_dirty = true;
                }
            }
        }

        // update grid texture if needed
        if (grid_dirty) {
            blit_grid(pixel_buf, grid_pixel_w, grid_pixel_h,
                      &train_images, offset, GRID_COLS, GRID_ROWS);
            SDL_UpdateTexture(grid_tex, NULL, pixel_buf, grid_pixel_w * sizeof(u32));
            grid_dirty = false;
        }

        // --- render ---
        SDL_SetRenderDrawColor(renderer, 25, 25, 25, 255);
        SDL_RenderClear(renderer);

        // MNIST grid on the left (not in spy mode)
        if (screen != SCREEN_SPY) {
            SDL_FRect grid_dst = {0, 0, grid_pixel_w, grid_pixel_h};
            SDL_RenderTexture(renderer, grid_tex, NULL, &grid_dst);
        }

        // right panel
        UILayout lay = ui_layout(win_w - panel_w + 16, 16, panel_w - 32, 6);

        ui_label(&ui, &lay, "MNIST Explorer", UI_WHITE);

        char buf[64];
        snprintf(buf, sizeof(buf), "showing %u-%u / %u",
                 offset, offset + GRID_COUNT - 1, train_images.count);
        ui_label(&ui, &lay, buf, UI_GRAY(160));
        ui_spacer(&lay, 8);

        if (screen == SCREEN_VIEW) {
            if (ui_button(&ui, "< prev", lay.x, lay.y, 120, 32)) {
                if (offset >= GRID_COUNT) offset -= GRID_COUNT;
                else offset = 0;
                grid_dirty = true;
            }
            if (ui_button(&ui, "next >", lay.x + 132, lay.y, 120, 32)) {
                if (offset + GRID_COUNT < train_images.count) {
                    offset += GRID_COUNT;
                    grid_dirty = true;
                }
            }
            ui_advance(&lay, 32);
            ui_spacer(&lay, 8);

            if (ui_lay_button(&ui, &lay, "Train", 36)) {
                screen = SCREEN_TRAINING;
                train_ctx = (TrainCtx){0};
                train_ctx.mlp = &mlp;
                train_ctx.images = &train_images;
                train_ctx.labels = &train_labels;
                train_ctx.test_images = &test_images;
                train_ctx.test_labels = &test_labels;
                train_ctx.lr = 0.001f;
                train_ctx.spy = &spy;
                pthread_create(&train_thread, NULL, train_thread_fn, &train_ctx);
                train_thread_active = true;
            }
            if (ui_lay_button(&ui, &lay, "Spy (random)", 36)) {
                arena_clear(spy.arena);
                spy_build_random(&spy, 100000);
                spy_build_pyramid(&spy);
                spy_level_idx = 0;
                spy_dirty = true;
                cam_x = cam_y = 0; cam_span = 1.0f;
                prev_screen = SCREEN_VIEW;
                screen = SCREEN_SPY;
            }
            ui_spacer(&lay, 8);

            ui_label(&ui, &lay, "arrows/space: browse", UI_GRAY(100));
            ui_label(&ui, &lay, "q/esc: quit", UI_GRAY(100));

        } else if (screen == SCREEN_TRAINING) {
            int s = atomic_load(&train_ctx.step);
            int ep = atomic_load(&train_ctx.epoch);
            int cor = atomic_load(&train_ctx.correct);
            float loss = atomic_load(&train_ctx.loss);
            float acc = s > 0 ? 100.0f * cor / s : 0;

            snprintf(buf, sizeof(buf), "epoch %d  step %d", ep, s);
            ui_label(&ui, &lay, buf, UI_RGB(100, 200, 100));

            snprintf(buf, sizeof(buf), "loss: %.4f", loss);
            ui_label(&ui, &lay, buf, UI_GRAY(160));

            snprintf(buf, sizeof(buf), "train acc: %.1f%%", acc);
            ui_label(&ui, &lay, buf, UI_GRAY(160));

            float ta = atomic_load(&train_ctx.test_acc);
            snprintf(buf, sizeof(buf), "test acc:  %.1f%%", ta);
            ui_label(&ui, &lay, buf, ta > 0 ? UI_WHITE : UI_GRAY(100));
            ui_spacer(&lay, 6);

            // test accuracy graph
            {
                float gx = lay.x, gy = lay.y, gw = lay.w, gh = 80;
                ui_rect(&ui, gx, gy, gw, gh, UI_GRAY(35));
                ui_rect_outline(&ui, gx, gy, gw, gh, UI_GRAY(60));

                int hc = atomic_load(&train_ctx.hist_count);
                int show = hc < ACC_HIST ? hc : ACC_HIST;
                int start = hc < ACC_HIST ? 0 : hc - ACC_HIST;

                if (show > 0) {
                    float ymin = 100, ymax = 0;
                    for (int i = 0; i < show; i++) {
                        float v = train_ctx.hist[(start + i) % ACC_HIST];
                        if (v < ymin) ymin = v;
                        if (v > ymax) ymax = v;
                    }
                    float pad = (ymax - ymin) * 0.15f;
                    if (pad < 2) pad = 2;
                    ymin -= pad; ymax += pad;
                    if (ymin < 0) ymin = 0;
                    if (ymax > 100) ymax = 100;
                    float yrange = ymax - ymin;
                    if (yrange < 1) yrange = 1;

                    snprintf(buf, sizeof(buf), "%.0f%%", ymax);
                    ui_text(&ui, buf, gx + 2, gy, UI_GRAY(100));
                    snprintf(buf, sizeof(buf), "%.0f%%", ymin);
                    ui_text(&ui, buf, gx + 2, gy + gh - 16, UI_GRAY(100));

                    for (int i = 0; i < show; i++) {
                        float v = train_ctx.hist[(start + i) % ACC_HIST];
                        float x0 = gx + (show > 1 ? (float)i / (show - 1) * gw : gw * 0.5f);
                        float y0 = gy + gh - (v - ymin) / yrange * gh;
                        ui_rect(&ui, x0 - 1, y0 - 1, 3, 3, UI_RGB(80, 180, 255));
                        if (i > 0) {
                            float vp = train_ctx.hist[(start + i - 1) % ACC_HIST];
                            float xp = gx + (float)(i - 1) / (show - 1) * gw;
                            float yp = gy + gh - (vp - ymin) / yrange * gh;
                            ui_line(&ui, xp, yp, x0, y0, UI_RGB(80, 180, 255));
                        }
                    }
                }
                ui_advance(&lay, gh);
            }

            if (ui_lay_button(&ui, &lay, "Spy (tape)", 32)) {
                atomic_store(&train_ctx.spy_state, 1);
            }
            // check if spy data is ready
            if (atomic_load(&train_ctx.spy_state) == 2) {
                atomic_store(&train_ctx.spy_state, 0);
                spy_level_idx = 0;
                spy_dirty = true;
                cam_x = cam_y = 0; cam_span = 1.0f;
                prev_screen = SCREEN_TRAINING;
                screen = SCREEN_SPY;
            }

            if (ui_lay_button(&ui, &lay, "Stop", 32)) {
                atomic_store(&train_ctx.stop, true);
                pthread_join(train_thread, NULL);
                train_thread_active = false;
                screen = SCREEN_VIEW;
            }

        } else if (screen == SCREEN_SPY) {
            // auto-select level: pick where visible cells ≈ SPY_CANVAS
            if (spy.num_levels > 0) {
                spy_level_idx = 0;
                for (u32 k = 0; k < spy.num_levels; k++) {
                    u32 vis = (u32)(cam_span * spy.levels[k].row_count + 0.5f);
                    if (vis <= SPY_CANVAS) { spy_level_idx = k; break; }
                    spy_level_idx = k;
                }
            }

            // update spy texture if needed
            static SpyRenderResult spy_res;
            if (spy_dirty && spy.num_levels > 0) {
                spy_res = spy_render_level(&spy, spy_level_idx, cam_x, cam_y, cam_span, spy_pixels);
                // recreate texture if size changed
                if (spy_res.tex_w != spy_tex_w || spy_res.tex_h != spy_tex_h) {
                    if (spy_tex) SDL_DestroyTexture(spy_tex);
                    spy_tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888,
                                                SDL_TEXTUREACCESS_STREAMING, spy_res.tex_w, spy_res.tex_h);
                    SDL_SetTextureScaleMode(spy_tex, SDL_SCALEMODE_NEAREST);
                    spy_tex_w = spy_res.tex_w;
                    spy_tex_h = spy_res.tex_h;
                }
                SDL_UpdateTexture(spy_tex, NULL, spy_pixels, spy_res.tex_w * sizeof(u32));
                spy_dirty = false;
            }

            // draw canvas — position quad according to snapped cell boundaries
            if (spy_tex) {
                SDL_FRect spy_dst = {spy_res.quad_x, spy_res.quad_y, spy_res.quad_w, spy_res.quad_h};
                SDL_RenderTexture(renderer, spy_tex, NULL, &spy_dst);
            }

            // sidebar
            SpyLevel *lv = &spy.levels[spy_level_idx];
            snprintf(buf, sizeof(buf), "level %u / %u", spy_level_idx, spy.num_levels - 1);
            ui_label(&ui, &lay, buf, UI_WHITE);

            snprintf(buf, sizeof(buf), "rows: %u  cols: %u", lv->row_count, lv->col_count);
            ui_label(&ui, &lay, buf, UI_GRAY(160));

            snprintf(buf, sizeof(buf), "nnz: %u", lv->nnz);
            ui_label(&ui, &lay, buf, UI_GRAY(160));

            snprintf(buf, sizeof(buf), "zoom: %.1fx", 1.0f / cam_span);
            ui_label(&ui, &lay, buf, UI_GRAY(160));
            ui_spacer(&lay, 8);

            ui_label(&ui, &lay, "scroll: zoom", UI_GRAY(100));
            ui_label(&ui, &lay, "drag: pan", UI_GRAY(100));
            ui_label(&ui, &lay, "home: reset", UI_GRAY(100));
            ui_label(&ui, &lay, "esc: back", UI_GRAY(100));
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    if (train_thread_active) {
        atomic_store(&train_ctx.stop, true);
        pthread_join(train_thread, NULL);
    }
    tape_destroy(&mlp.tape);

    arena_destroy(spy.arena);
    ui_destroy(&ui);
    if (spy_tex) SDL_DestroyTexture(spy_tex);
    SDL_DestroyTexture(grid_tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    arena_destroy(arena);
    return 0;
}
