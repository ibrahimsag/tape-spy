#define ARENA_IMPLEMENTATION
#include "arena.h"
#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

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

    if (!train_images.pixels || !train_labels.labels) {
        fprintf(stderr, "failed to load MNIST data\n");
        return 1;
    }
    printf("loaded %u images (%ux%u), %u labels\n",
           train_images.count, train_images.rows, train_images.cols,
           train_labels.count);

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
    int win_h = grid_pixel_h;

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
                // TODO: kick off training thread
            }
            py += 52;

            ui_text(&ui, "arrows/space: browse", px, py, UI_GRAY(100));
            py += 20;
            ui_text(&ui, "q/esc: quit", px, py, UI_GRAY(100));

        } else if (screen == SCREEN_TRAINING) {
            ui_text(&ui, "Training...", px, py, UI_RGB(100, 200, 100));
            py += 28;

            // TODO: show live loss/accuracy from train thread
            ui_text(&ui, "loss: ---", px, py, UI_GRAY(160));
            py += 22;
            ui_text(&ui, "accuracy: ---", px, py, UI_GRAY(160));
            py += 40;

            if (ui_button(&ui, "Back", px, py, 252, 32)) {
                screen = SCREEN_VIEW;
            }
        }

        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    ui_destroy(&ui);
    SDL_DestroyTexture(grid_tex);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    arena_destroy(arena);
    return 0;
}
