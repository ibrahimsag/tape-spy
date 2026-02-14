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

// --- MLP (784 -> 128 -> 10) ---

#define MLP_IN   784
#define MLP_HID1 128
#define MLP_HID2 64
#define MLP_OUT  10

typedef struct {
    u32 start;    // first weight param index in tape
    u32 in_dim;
    u32 out_dim;
    // weights[i][j] at: start + i * in_dim + j
    // biases[i] at:     start + out_dim * in_dim + i
} Layer;

typedef struct {
    Tape tape;
    Layer l1, l2, l3;
    u32 logits[MLP_OUT]; // filled by mlp_forward
} MLP;

static MLP mlp_create(void) {
    MLP m;
    m.tape = tape_create(MiB(64));

    // Layer 1: 784 -> 128 (He init)
    m.l1 = (Layer){ .start = m.tape.count, .in_dim = MLP_IN, .out_dim = MLP_HID1 };
    f32 s1 = sqrtf(2.0f / MLP_IN);
    for (u32 i = 0; i < MLP_HID1 * MLP_IN; i++) tape_param(&m.tape, rng_normal() * s1);
    for (u32 i = 0; i < MLP_HID1; i++) tape_param(&m.tape, 0.0f);

    // Layer 2: 128 -> 64
    m.l2 = (Layer){ .start = m.tape.count, .in_dim = MLP_HID1, .out_dim = MLP_HID2 };
    f32 s2 = sqrtf(2.0f / MLP_HID1);
    for (u32 i = 0; i < MLP_HID2 * MLP_HID1; i++) tape_param(&m.tape, rng_normal() * s2);
    for (u32 i = 0; i < MLP_HID2; i++) tape_param(&m.tape, 0.0f);

    // Layer 3: 64 -> 10
    m.l3 = (Layer){ .start = m.tape.count, .in_dim = MLP_HID2, .out_dim = MLP_OUT };
    f32 s3 = sqrtf(2.0f / MLP_HID2);
    for (u32 i = 0; i < MLP_OUT * MLP_HID2; i++) tape_param(&m.tape, rng_normal() * s3);
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

// Forward pass: returns loss node index, fills m->logits
static u32 mlp_forward(MLP *m, u8 *pixels, u8 label) {
    Tape *t = &m->tape;

    // Input leaves (normalize 0-255 -> 0-1)
    u32 inp[MLP_IN];
    for (u32 i = 0; i < MLP_IN; i++)
        inp[i] = tape_leaf(t, (f32)pixels[i] / 255.0f);

    // Layer 1 + ReLU
    u32 h1[MLP_HID1];
    linear_fwd(t, &m->l1, inp, h1);
    for (u32 i = 0; i < MLP_HID1; i++) h1[i] = val_relu(t, h1[i]);

    // Layer 2 + ReLU
    u32 h2[MLP_HID2];
    linear_fwd(t, &m->l2, h1, h2);
    for (u32 i = 0; i < MLP_HID2; i++) h2[i] = val_relu(t, h2[i]);

    // Layer 3 (logits)
    linear_fwd(t, &m->l3, h2, m->logits);

    // Cross-entropy via log-softmax (numerically stable)
    f32 max_v = tape_data(t, m->logits[0]);
    for (u32 i = 1; i < MLP_OUT; i++) {
        f32 v = tape_data(t, m->logits[i]);
        if (v > max_v) max_v = v;
    }
    u32 mx = tape_leaf(t, max_v);

    // sum of exp(logit - max)
    u32 se = val_exp(t, val_sub(t, m->logits[0], mx));
    for (u32 i = 1; i < MLP_OUT; i++)
        se = val_add(t, se, val_exp(t, val_sub(t, m->logits[i], mx)));

    // loss = log(sum_exp) - (logit[label] - max) = -log_softmax(label)
    return val_sub(t, val_log(t, se), val_sub(t, m->logits[label], mx));
}

static void sgd_step(Tape *t, f32 lr) {
    for (u32 i = 0; i < t->perm_count; i++)
        t->base[i].data -= lr * t->base[i].grad;
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

    for (int epoch = 0; !atomic_load(&ctx->stop); epoch++) {
        atomic_store(&ctx->epoch, epoch);
        atomic_store(&ctx->step, 0);
        atomic_store(&ctx->correct, 0);
        for (u32 i = 0; i < ctx->images->count && !atomic_load(&ctx->stop); i++) {
            u8 *px = ctx->images->pixels + (u64)i * 784;
            u8 label = ctx->labels->labels[i];

            tape_reset(&ctx->mlp->tape);
            u32 loss_idx = mlp_forward(ctx->mlp, px, label);
            tape_backward(&ctx->mlp->tape, loss_idx);
            sgd_step(&ctx->mlp->tape, ctx->lr);

            if (mlp_predict(ctx->mlp) == label)
                atomic_fetch_add(&ctx->correct, 1);

            atomic_store(&ctx->step, (int)i + 1);
            atomic_store(&ctx->loss, tape_data(&ctx->mlp->tape, loss_idx));

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
    SCREEN_RESULTS,
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

    if (!SDL_Init(SDL_INIT_VIDEO)) {
        fprintf(stderr, "SDL_Init failed: %s\n", SDL_GetError());
        return 1;
    }

    int img_w = train_images.cols;
    int img_h = train_images.rows;
    int panel_w = 280;
    int grid_pixel_w = img_w * GRID_COLS;
    int grid_pixel_h = img_h * GRID_ROWS;
    int win_w = grid_pixel_w + panel_w;
    int win_h = grid_pixel_h > 400 ? grid_pixel_h : 400;

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
                if (event.key.key == SDLK_ESCAPE || event.key.key == SDLK_Q) {
                    running = false;
                }
                if (screen == SCREEN_VIEW) {
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

        // MNIST grid on the left
        SDL_FRect grid_dst = {0, 0, grid_pixel_w, grid_pixel_h};
        SDL_RenderTexture(renderer, grid_tex, NULL, &grid_dst);

        // right panel
        float px = grid_pixel_w + 16;
        float py = 16;

        ui_text(&ui, "MNIST Explorer", px, py, UI_WHITE);
        py += 28;

        char buf[64];
        snprintf(buf, sizeof(buf), "showing %u-%u / %u",
                 offset, offset + GRID_COUNT - 1, train_images.count);
        ui_text(&ui, buf, px, py, UI_GRAY(160));
        py += 36;

        if (screen == SCREEN_VIEW) {
            if (ui_button(&ui, "< prev", px, py, 120, 32)) {
                if (offset >= GRID_COUNT) offset -= GRID_COUNT;
                else offset = 0;
                grid_dirty = true;
            }
            if (ui_button(&ui, "next >", px + 132, py, 120, 32)) {
                if (offset + GRID_COUNT < train_images.count) {
                    offset += GRID_COUNT;
                    grid_dirty = true;
                }
            }
            py += 48;

            if (ui_button(&ui, "Train", px, py, 252, 36)) {
                screen = SCREEN_TRAINING;
                train_ctx = (TrainCtx){0};
                train_ctx.mlp = &mlp;
                train_ctx.images = &train_images;
                train_ctx.labels = &train_labels;
                train_ctx.test_images = &test_images;
                train_ctx.test_labels = &test_labels;
                train_ctx.lr = 0.001f;
                pthread_create(&train_thread, NULL, train_thread_fn, &train_ctx);
                train_thread_active = true;
            }
            py += 52;

            ui_text(&ui, "arrows/space: browse", px, py, UI_GRAY(100));
            py += 20;
            ui_text(&ui, "q/esc: quit", px, py, UI_GRAY(100));

        } else if (screen == SCREEN_TRAINING) {
            int s = atomic_load(&train_ctx.step);
            int ep = atomic_load(&train_ctx.epoch);
            int cor = atomic_load(&train_ctx.correct);
            float loss = atomic_load(&train_ctx.loss);
            float acc = s > 0 ? 100.0f * cor / s : 0;

            snprintf(buf, sizeof(buf), "epoch %d  step %d", ep, s);
            ui_text(&ui, buf, px, py, UI_RGB(100, 200, 100));
            py += 22;

            snprintf(buf, sizeof(buf), "loss: %.4f", loss);
            ui_text(&ui, buf, px, py, UI_GRAY(160));
            py += 22;

            snprintf(buf, sizeof(buf), "train acc: %.1f%%", acc);
            ui_text(&ui, buf, px, py, UI_GRAY(160));
            py += 22;

            float ta = atomic_load(&train_ctx.test_acc);
            snprintf(buf, sizeof(buf), "test acc:  %.1f%%", ta);
            ui_text(&ui, buf, px, py, ta > 0 ? UI_WHITE : UI_GRAY(100));
            py += 28;

            // test accuracy graph
            {
                float gx = px, gy = py, gw = 240, gh = 80;
                ui_rect(&ui, gx, gy, gw, gh, UI_GRAY(35));
                ui_rect_outline(&ui, gx, gy, gw, gh, UI_GRAY(60));

                int hc = atomic_load(&train_ctx.hist_count);
                int show = hc < ACC_HIST ? hc : ACC_HIST;
                int start = hc < ACC_HIST ? 0 : hc - ACC_HIST;

                if (show > 0) {
                    // find y range
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

                    // y-axis labels
                    snprintf(buf, sizeof(buf), "%.0f%%", ymax);
                    ui_text(&ui, buf, gx + 2, gy, UI_GRAY(100));
                    snprintf(buf, sizeof(buf), "%.0f%%", ymin);
                    ui_text(&ui, buf, gx + 2, gy + gh - 16, UI_GRAY(100));

                    // line graph
                    for (int i = 0; i < show; i++) {
                        float v = train_ctx.hist[(start + i) % ACC_HIST];
                        float x0 = gx + (show > 1 ? (float)i / (show - 1) * gw : gw * 0.5f);
                        float y0 = gy + gh - (v - ymin) / yrange * gh;
                        // dot
                        ui_rect(&ui, x0 - 1, y0 - 1, 3, 3, UI_RGB(80, 180, 255));
                        // line to previous
                        if (i > 0) {
                            float vp = train_ctx.hist[(start + i - 1) % ACC_HIST];
                            float xp = gx + (float)(i - 1) / (show - 1) * gw;
                            float yp = gy + gh - (vp - ymin) / yrange * gh;
                            ui_line(&ui, xp, yp, x0, y0, UI_RGB(80, 180, 255));
                        }
                    }
                }
                py += gh + 8;
            }

            if (ui_button(&ui, "Stop", px, py, 252, 32)) {
                atomic_store(&train_ctx.stop, true);
                pthread_join(train_thread, NULL);
                train_thread_active = false;
                screen = SCREEN_VIEW;
            }
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    if (train_thread_active) {
        atomic_store(&train_ctx.stop, true);
        pthread_join(train_thread, NULL);
    }
    tape_destroy(&mlp.tape);

    ui_destroy(&ui);
    SDL_DestroyTexture(grid_tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    arena_destroy(arena);
    return 0;
}
