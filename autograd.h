/*
 * autograd.h — stb-style scalar autograd with arena-backed tape
 *
 * Usage:
 *   #define AUTOGRAD_IMPLEMENTATION
 *   #include "autograd.h"
 *
 * in exactly one C file. Requires arena.h.
 *
 * The tape is a flat array of Nodes in creation order.
 * Creation order = topological order, so backward is a reverse linear scan.
 */

#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include "arena.h"
#include <math.h>

/* ---- node ---- */

typedef struct {
    f32 data;
    f32 grad;
    u32 children[2];    // indices into tape
    f32 local_grads[2]; // d(this)/d(child), set at forward time
    u32 arity;          // 0 = leaf, 1 = unary, 2 = binary
    u32 _pad;
} Node; // 32 bytes

/* ---- tape ---- */

typedef struct {
    mem_arena *arena;
    Node *base;         // start of node array in arena
    u32 count;          // total nodes so far
    u32 perm_count;     // parameter nodes (survive reset)
} Tape;

/* ---- lifecycle ---- */

Tape  tape_create(u64 reserve_size);
void  tape_destroy(Tape *t);
void  tape_reset(Tape *t);       // free computation nodes, zero param grads

/* ---- leaf creation ---- */

u32 tape_param(Tape *t, f32 data);  // permanent parameter node
u32 tape_leaf(Tape *t, f32 data);   // ephemeral constant node

/* ---- operations ---- */

u32 val_add(Tape *t, u32 a, u32 b);
u32 val_sub(Tape *t, u32 a, u32 b);
u32 val_mul(Tape *t, u32 a, u32 b);
u32 val_neg(Tape *t, u32 a);
u32 val_pow(Tape *t, u32 a, f32 n);  // n is a plain float
u32 val_log(Tape *t, u32 a);
u32 val_exp(Tape *t, u32 a);
u32 val_relu(Tape *t, u32 a);
u32 val_div(Tape *t, u32 a, u32 b);  // = mul(a, pow(b, -1))

/* ---- backward ---- */

void tape_backward(Tape *t, u32 loss_idx);

/* ---- accessors ---- */

static inline f32  tape_data(Tape *t, u32 i) { return t->base[i].data; }
static inline f32  tape_grad(Tape *t, u32 i) { return t->base[i].grad; }
static inline void tape_set_data(Tape *t, u32 i, f32 v) { t->base[i].data = v; }

#endif /* AUTOGRAD_H */

/* ================================================================
 * Implementation
 * ================================================================ */

#ifdef AUTOGRAD_IMPLEMENTATION

/* ---- internal: push a node onto the tape ---- */

static u32 _tape_push(Tape *t, f32 data, u8 arity,
                       u32 c0, u32 c1, f32 lg0, f32 lg1) {
    Node *n = PUSH_STRUCT_NZ(t->arena, Node);
    n->data = data;
    n->grad = 0.0f;
    n->arity = arity;
    n->children[0] = c0;
    n->children[1] = c1;
    n->local_grads[0] = lg0;
    n->local_grads[1] = lg1;
    return t->count++;
}

/* ---- lifecycle ---- */

Tape tape_create(u64 reserve_size) {
    Tape t;
    t.arena = arena_create(reserve_size, MiB(1));
    t.base = (Node *)((u8 *)t.arena + ARENA_BASE_POS);
    t.count = 0;
    t.perm_count = 0;
    return t;
}

void tape_destroy(Tape *t) {
    arena_destroy(t->arena);
}

void tape_reset(Tape *t) {
    u64 perm_end = ARENA_BASE_POS + (u64)t->perm_count * sizeof(Node);
    arena_pop_to(t->arena, perm_end);
    t->count = t->perm_count;
}

void tape_zero_grad(Tape *t) {
    for (u32 i = 0; i < t->perm_count; i++)
        t->base[i].grad = 0.0f;
}

/* ---- leaf creation ---- */

u32 tape_param(Tape *t, f32 data) {
    u32 idx = _tape_push(t, data, 0, 0, 0, 0, 0);
    t->perm_count = t->count; // all params must be created before any computation
    return idx;
}

u32 tape_leaf(Tape *t, f32 data) {
    return _tape_push(t, data, 0, 0, 0, 0, 0);
}

/* ---- operations ---- */

u32 val_add(Tape *t, u32 a, u32 b) {
    f32 da = t->base[a].data, db = t->base[b].data;
    return _tape_push(t, da + db, 2, a, b, 1.0f, 1.0f);
}

u32 val_sub(Tape *t, u32 a, u32 b) {
    f32 da = t->base[a].data, db = t->base[b].data;
    return _tape_push(t, da - db, 2, a, b, 1.0f, -1.0f);
}

u32 val_mul(Tape *t, u32 a, u32 b) {
    f32 da = t->base[a].data, db = t->base[b].data;
    return _tape_push(t, da * db, 2, a, b, db, da);
}

u32 val_neg(Tape *t, u32 a) {
    f32 da = t->base[a].data;
    return _tape_push(t, -da, 1, a, 0, -1.0f, 0.0f);
}

u32 val_pow(Tape *t, u32 a, f32 n) {
    f32 da = t->base[a].data;
    return _tape_push(t, powf(da, n), 1, a, 0,
                      n * powf(da, n - 1.0f), 0.0f);
}

u32 val_log(Tape *t, u32 a) {
    f32 da = t->base[a].data;
    return _tape_push(t, logf(da), 1, a, 0, 1.0f / da, 0.0f);
}

u32 val_exp(Tape *t, u32 a) {
    f32 da = t->base[a].data;
    f32 e = expf(da);
    return _tape_push(t, e, 1, a, 0, e, 0.0f);
}

u32 val_relu(Tape *t, u32 a) {
    f32 da = t->base[a].data;
    return _tape_push(t, da > 0 ? da : 0, 1, a, 0,
                      da > 0 ? 1.0f : 0.0f, 0.0f);
}

u32 val_div(Tape *t, u32 a, u32 b) {
    u32 inv_b = val_pow(t, b, -1.0f);
    return val_mul(t, a, inv_b);
}

/* ---- backward ---- */

void tape_backward(Tape *t, u32 loss_idx) {
    t->base[loss_idx].grad = 1.0f;
    for (i32 i = (i32)loss_idx; i >= 0; i--) {
        Node *v = &t->base[i];
        for (u32 c = 0; c < v->arity; c++) {
            t->base[v->children[c]].grad += v->local_grads[c] * v->grad;
        }
    }
}

#endif /* AUTOGRAD_IMPLEMENTATION */
