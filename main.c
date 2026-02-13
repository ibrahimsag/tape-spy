#define ARENA_IMPLEMENTATION
#include "arena.h"
#define AUTOGRAD_IMPLEMENTATION
#include "autograd.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_main.h>

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

// --- rendering ---

#define DISPLAY_SCALE 2
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
        // z = x*y + x  =>  dz/dx = y + 1 = 5,  dz/dy = x = 3
        u32 xy = val_mul(&tape, x, y);
        u32 z  = val_add(&tape, xy, x);
        tape_backward(&tape, z);
        printf("autograd test: z = x*y + x, x=3, y=4\n");
        printf("  z    = %.1f (expect 15.0)\n", tape_data(&tape, z));
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
    int win_w = img_w * GRID_COLS * DISPLAY_SCALE;
    int win_h = img_h * GRID_ROWS * DISPLAY_SCALE;

    SDL_Window *window = SDL_CreateWindow("MNIST", win_w, win_h, 0);
    if (!window) {
        fprintf(stderr, "SDL_CreateWindow failed: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Renderer *renderer = SDL_CreateRenderer(window, NULL);
    if (!renderer) {
        fprintf(stderr, "SDL_CreateRenderer failed: %s\n", SDL_GetError());
        return 1;
    }

    // texture for the full grid
    int tex_w = img_w * GRID_COLS;
    int tex_h = img_h * GRID_ROWS;
    SDL_Texture *texture = SDL_CreateTexture(
        renderer, SDL_PIXELFORMAT_ABGR8888,
        SDL_TEXTUREACCESS_STREAMING, tex_w, tex_h
    );

    u32 *pixel_buf = PUSH_ARRAY(arena, u32, tex_w * tex_h);
    u32 offset = 0;

    // blit grid of images into pixel buffer
    for (int gy = 0; gy < GRID_ROWS; gy++) {
        for (int gx = 0; gx < GRID_COLS; gx++) {
            u32 idx = offset + gy * GRID_COLS + gx;
            if (idx >= train_images.count) break;
            u8 *src = train_images.pixels + (u64)idx * img_w * img_h;
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

    bool running = true;
    while (running) {
        SDL_Event event;
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_EVENT_QUIT) {
                running = false;
            }
            if (event.type == SDL_EVENT_KEY_DOWN) {
                if (event.key.key == SDLK_ESCAPE || event.key.key == SDLK_Q) {
                    running = false;
                }
                if (event.key.key == SDLK_RIGHT || event.key.key == SDLK_SPACE) {
                    if (offset + GRID_COUNT < train_images.count) offset += GRID_COUNT;
                }
                if (event.key.key == SDLK_LEFT) {
                    if (offset >= GRID_COUNT) offset -= GRID_COUNT;
                    else offset = 0;
                }

                // re-blit
                memset(pixel_buf, 0, sizeof(u32) * tex_w * tex_h);
                for (int gy = 0; gy < GRID_ROWS; gy++) {
                    for (int gx = 0; gx < GRID_COLS; gx++) {
                        u32 idx = offset + gy * GRID_COLS + gx;
                        if (idx >= train_images.count) break;
                        u8 *src = train_images.pixels + (u64)idx * img_w * img_h;
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
        }

        SDL_UpdateTexture(texture, NULL, pixel_buf, tex_w * sizeof(u32));
        SDL_RenderClear(renderer);
        SDL_RenderTexture(renderer, texture, NULL, NULL);
        SDL_RenderPresent(renderer);
        SDL_Delay(16);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
    arena_destroy(arena);
    return 0;
}
