/*
 * ui.h — minimal immediate-mode UI on SDL_Renderer + stb_truetype
 *
 * Usage:
 *   #define UI_IMPLEMENTATION
 *   #include "ui.h"
 *
 * Requires: SDL3, stb_truetype.h in include path, a .ttf font file.
 */

#ifndef UI_H
#define UI_H

#include <SDL3/SDL.h>
#include "stb_truetype.h"

/* ---- colors ---- */

typedef struct { uint8_t r, g, b, a; } UIColor;

#define UI_RGB(r,g,b)    (UIColor){(r),(g),(b),255}
#define UI_RGBA(r,g,b,a) (UIColor){(r),(g),(b),(a)}
#define UI_WHITE  UI_RGB(255, 255, 255)
#define UI_GRAY(v) UI_RGB((v),(v),(v))
#define UI_BLACK  UI_RGB(0, 0, 0)

/* ---- context ---- */

#define UI_ATLAS_W 1024
#define UI_ATLAS_H 1024

typedef struct {
    SDL_Renderer *renderer;

    /* font */
    SDL_Texture *font_tex;
    stbtt_bakedchar cdata[96]; /* ASCII 32..127 */
    float font_size;    /* logical size (what callers use) */
    float bake_size;    /* actual bake size (font_size * dpi_scale) */
    float dpi_scale;

    /* input (updated per frame) */
    float mouse_x, mouse_y;
    bool mouse_down;
    bool mouse_pressed;   /* just went down this frame */
    bool mouse_released;  /* just went up this frame */

    /* imgui-style interaction */
    uint32_t hot;
    uint32_t active;
} UICtx;

/* ---- API ---- */

bool  ui_init(UICtx *ctx, SDL_Renderer *renderer, const char *font_path, float font_size, float dpi_scale);
void  ui_destroy(UICtx *ctx);

/* call once per frame before any ui_ draw calls */
void  ui_begin(UICtx *ctx);
/* feed SDL events (call for each event) */
void  ui_event(UICtx *ctx, SDL_Event *e);

/* text */
void  ui_text(UICtx *ctx, const char *text, float x, float y, UIColor color);
float ui_measure(UICtx *ctx, const char *text);

/* widgets — return true on click */
bool  ui_button(UICtx *ctx, const char *label, float x, float y, float w, float h);

/* primitives */
void  ui_rect(UICtx *ctx, float x, float y, float w, float h, UIColor color);
void  ui_rect_outline(UICtx *ctx, float x, float y, float w, float h, UIColor color);
void  ui_line(UICtx *ctx, float x1, float y1, float x2, float y2, UIColor color);

/* heatmap cell — rect with brightness mapped from value 0..1 */
void  ui_heat(UICtx *ctx, float x, float y, float w, float h, float value);

/* ---- layout cursor ---- */

typedef struct {
    float x, y;       /* current cursor position */
    float w;           /* available width */
    float spacing;     /* default gap between rows */
} UILayout;

/* start a vertical layout at (x,y) with width w and row spacing */
static inline UILayout ui_layout(float x, float y, float w, float spacing) {
    return (UILayout){ x, y, w, spacing };
}

/* advance cursor by h + spacing */
static inline void ui_advance(UILayout *lay, float h) {
    lay->y += h + lay->spacing;
}

/* extra vertical gap (beyond default spacing) */
static inline void ui_spacer(UILayout *lay, float extra) {
    lay->y += extra;
}

/* text at cursor, advance by font_size */
void  ui_label(UICtx *ctx, UILayout *lay, const char *text, UIColor color);

/* full-width button at cursor, advance by h */
bool  ui_lay_button(UICtx *ctx, UILayout *lay, const char *label, float h);

#endif /* UI_H */

/* ================================================================ */

#ifdef UI_IMPLEMENTATION

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

/* ---- helpers ---- */

static uint32_t _ui_hash(const char *s) {
    uint32_t h = 5381;
    while (*s) h = ((h << 5) + h) + (uint8_t)*s++;
    return h ? h : 1; /* never return 0, 0 = no element */
}

static void _ui_set_color(SDL_Renderer *r, UIColor c) {
    SDL_SetRenderDrawColor(r, c.r, c.g, c.b, c.a);
}

/* ---- lifecycle ---- */

bool ui_init(UICtx *ctx, SDL_Renderer *renderer, const char *font_path, float font_size, float dpi_scale) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->renderer = renderer;
    ctx->font_size = font_size;
    ctx->dpi_scale = dpi_scale;
    ctx->bake_size = font_size * dpi_scale * 2.0f;

    /* load font file */
    FILE *f = fopen(font_path, "rb");
    if (!f) { fprintf(stderr, "ui: can't open font %s\n", font_path); return false; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char *font_data = malloc(fsize);
    fread(font_data, 1, fsize, f);
    fclose(f);

    /* bake font bitmap */
    unsigned char *bitmap = malloc(UI_ATLAS_W * UI_ATLAS_H);
    stbtt_BakeFontBitmap(font_data, 0, ctx->bake_size, bitmap, UI_ATLAS_W, UI_ATLAS_H, 32, 96, ctx->cdata);
    free(font_data);

    /* convert alpha bitmap to RGBA */
    uint32_t *rgba = malloc(UI_ATLAS_W * UI_ATLAS_H * 4);
    for (int i = 0; i < UI_ATLAS_W * UI_ATLAS_H; i++) {
        rgba[i] = 0x00FFFFFF | ((uint32_t)bitmap[i] << 24);
    }
    free(bitmap);

    /* upload to SDL texture */
    ctx->font_tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_ABGR8888,
                                       SDL_TEXTUREACCESS_STATIC, UI_ATLAS_W, UI_ATLAS_H);
    SDL_UpdateTexture(ctx->font_tex, NULL, rgba, UI_ATLAS_W * 4);
    SDL_SetTextureBlendMode(ctx->font_tex, SDL_BLENDMODE_BLEND);
    SDL_SetTextureScaleMode(ctx->font_tex, SDL_SCALEMODE_LINEAR);
    free(rgba);

    return true;
}

void ui_destroy(UICtx *ctx) {
    if (ctx->font_tex) SDL_DestroyTexture(ctx->font_tex);
}

/* ---- frame ---- */

void ui_begin(UICtx *ctx) {
    ctx->hot = 0;
    ctx->mouse_pressed = false;
    ctx->mouse_released = false;
}

void ui_event(UICtx *ctx, SDL_Event *e) {
    if (e->type == SDL_EVENT_MOUSE_MOTION) {
        ctx->mouse_x = e->motion.x;
        ctx->mouse_y = e->motion.y;
    }
    if (e->type == SDL_EVENT_MOUSE_BUTTON_DOWN && e->button.button == SDL_BUTTON_LEFT) {
        ctx->mouse_down = true;
        ctx->mouse_pressed = true;
    }
    if (e->type == SDL_EVENT_MOUSE_BUTTON_UP && e->button.button == SDL_BUTTON_LEFT) {
        ctx->mouse_down = false;
        ctx->mouse_released = true;
    }
}

/* ---- text ---- */

void ui_text(UICtx *ctx, const char *text, float x, float y, UIColor color) {
    SDL_SetTextureColorMod(ctx->font_tex, color.r, color.g, color.b);
    SDL_SetTextureAlphaMod(ctx->font_tex, color.a);

    float s = ctx->font_size / ctx->bake_size;
    /* stbtt works in bake_size coordinates, we scale to logical */
    float cx = x / s, cy = (y + ctx->font_size) / s;
    while (*text) {
        if (*text >= 32 && (unsigned char)*text < 128) {
            stbtt_aligned_quad q;
            stbtt_GetBakedQuad(ctx->cdata, UI_ATLAS_W, UI_ATLAS_H,
                               *text - 32, &cx, &cy, &q, 1);
            SDL_FRect src = {q.s0 * UI_ATLAS_W, q.t0 * UI_ATLAS_H,
                             (q.s1 - q.s0) * UI_ATLAS_W, (q.t1 - q.t0) * UI_ATLAS_H};
            SDL_FRect dst = {q.x0 * s, q.y0 * s, (q.x1 - q.x0) * s, (q.y1 - q.y0) * s};
            SDL_RenderTexture(ctx->renderer, ctx->font_tex, &src, &dst);
        }
        text++;
    }
}

float ui_measure(UICtx *ctx, const char *text) {
    float cx = 0, cy = 0;
    while (*text) {
        if (*text >= 32 && (unsigned char)*text < 128) {
            stbtt_aligned_quad q;
            stbtt_GetBakedQuad(ctx->cdata, UI_ATLAS_W, UI_ATLAS_H,
                               *text - 32, &cx, &cy, &q, 1);
        }
        text++;
    }
    return cx * ctx->font_size / ctx->bake_size;
}

/* ---- widgets ---- */

bool ui_button(UICtx *ctx, const char *label, float x, float y, float w, float h) {
    uint32_t id = _ui_hash(label);
    bool hovered = ctx->mouse_x >= x && ctx->mouse_x < x + w &&
                   ctx->mouse_y >= y && ctx->mouse_y < y + h;
    bool clicked = false;

    if (hovered) {
        ctx->hot = id;
        if (ctx->mouse_pressed && ctx->active == 0) {
            ctx->active = id;
        }
    }
    if (ctx->active == id) {
        if (ctx->mouse_released) {
            if (hovered) clicked = true;
            ctx->active = 0;
        }
    }

    /* draw */
    uint8_t bg = 40;
    if (hovered) bg = (ctx->active == id) ? 80 : 60;
    ui_rect(ctx, x, y, w, h, UI_GRAY(bg));
    ui_rect_outline(ctx, x, y, w, h, UI_GRAY(140));

    /* center text */
    float tw = ui_measure(ctx, label);
    float tx = x + (w - tw) * 0.5f;
    float ty = y + (h - ctx->font_size) * 0.5f;
    ui_text(ctx, label, tx, ty, UI_WHITE);

    return clicked;
}

/* ---- primitives ---- */

void ui_rect(UICtx *ctx, float x, float y, float w, float h, UIColor color) {
    _ui_set_color(ctx->renderer, color);
    SDL_FRect r = {x, y, w, h};
    SDL_RenderFillRect(ctx->renderer, &r);
}

void ui_rect_outline(UICtx *ctx, float x, float y, float w, float h, UIColor color) {
    _ui_set_color(ctx->renderer, color);
    SDL_FRect r = {x, y, w, h};
    SDL_RenderRect(ctx->renderer, &r);
}

void ui_line(UICtx *ctx, float x1, float y1, float x2, float y2, UIColor color) {
    _ui_set_color(ctx->renderer, color);
    SDL_RenderLine(ctx->renderer, x1, y1, x2, y2);
}

void ui_heat(UICtx *ctx, float x, float y, float w, float h, float value) {
    if (value < 0) value = 0;
    if (value > 1) value = 1;
    uint8_t v = (uint8_t)(value * 255.0f);
    ui_rect(ctx, x, y, w, h, UI_RGB(v, v, v));
}

/* ---- layout widgets ---- */

void ui_label(UICtx *ctx, UILayout *lay, const char *text, UIColor color) {
    ui_text(ctx, text, lay->x, lay->y, color);
    ui_advance(lay, ctx->font_size);
}

bool ui_lay_button(UICtx *ctx, UILayout *lay, const char *label, float h) {
    bool clicked = ui_button(ctx, label, lay->x, lay->y, lay->w, h);
    ui_advance(lay, h);
    return clicked;
}

#endif /* UI_IMPLEMENTATION */
