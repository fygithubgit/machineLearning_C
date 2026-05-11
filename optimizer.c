#include "optimizer.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static f32 compute_total_grad_norm(matrix** grads, u32 num_grads) {
    f32 total = 0.0f;
    for (u32 i = 0; i < num_grads; i++) {
        if (grads[i]) {
            total += mat_sum_of_squares(grads[i]);
        }
    }
    return sqrtf(total);
}

void optimizer_clip_grad_norm(matrix** grads, u32 num_grads, f32 max_norm) {
    if (!grads || num_grads == 0 || max_norm <= 0.0f) return;
    f32 norm = compute_total_grad_norm(grads, num_grads);
    if (norm > max_norm) {
        f32 scale = max_norm / norm;
        for (u32 i = 0; i < num_grads; i++) {
            if (grads[i]) {
                mat_scale_inplace(grads[i], scale);
            }
        }
    }
}

optimizer* optimizer_sgd_create(mem_arena* arena, f32 learning_rate) {
    if (!arena || learning_rate <= 0.0f) return NULL;
    
    optimizer* opt = ARENA_ALLOC_STRUCT(arena, optimizer);
    if (!opt) return NULL;
    
    memset(opt, 0, sizeof(optimizer));
    opt->type = OPT_SGD;
    opt->learning_rate = learning_rate;
    opt->arena = arena;
    
    return opt;
}

optimizer* optimizer_sgd_momentum_create(mem_arena* arena, f32 lr, f32 momentum) {
    if (!arena || lr <= 0.0f || momentum < 0.0f || momentum >= 1.0f) return NULL;
    optimizer* opt = (optimizer*)mem_arena_alloc(arena, sizeof(optimizer), ARENA_DEFAULT_ALIGNMENT);
    if (!opt) return NULL;
    memset(opt, 0, sizeof(optimizer));
    opt->type = OPT_SGD_MOMENTUM;
    opt->learning_rate = lr;
    opt->momentum = momentum;
    opt->use_momentum = true;
    opt->arena = arena;
    return opt;
}

void optimizer_step(optimizer* opt, matrix** params, matrix** grads, u32 num_params) {
    if (!opt || !params || !grads || num_params == 0) return;

    // 首次调用：分配动量缓存
    if (opt->use_momentum && opt->velocities == NULL) {
        // 使用新的宏，包含alignment参数
        opt->velocities = (matrix**)mem_arena_alloc(
            opt->arena, 
            sizeof(matrix*) * num_params, 
            ARENA_DEFAULT_ALIGNMENT
        );
        if (!opt->velocities) return;
        memset(opt->velocities, 0, sizeof(matrix*) * num_params);
    }

    for (u32 i = 0; i < num_params; i++) {
        matrix* p = params[i];
        matrix* g = grads[i];
        if (!p || !g) continue;

        if (opt->use_momentum) {
            matrix* v = opt->velocities[i];
            if (!v) {
                // 首次为该参数分配 velocity（形状同参数）
                v = mat_create(opt->arena, p->rows, p->cols, MAT_DEFAULT_OPTS);
                mat_fill(v, 0.0f);
                opt->velocities[i] = v;
            }
            // v = momentum * v + grad
            mat_scale_inplace(v, opt->momentum);
            mat_add_inplace(v, g);
            // p = p - lr * v
            mat_axpy_inplace(p, -opt->learning_rate, v);
        } else {
            // p = p - lr * grad
            mat_axpy_inplace(p, -opt->learning_rate, g);
        }
    }
}