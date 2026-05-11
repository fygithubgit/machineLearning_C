#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "matrix.h"
#include <stdbool.h>

typedef enum {
    OPT_SGD,
    OPT_SGD_MOMENTUM
} optimizer_type;

typedef struct optimizer {
    optimizer_type type;
    f32 learning_rate;
    f32 momentum;
    bool use_momentum;
    matrix** velocities;   // 同 params 形状
    u32 num_params;
    mem_arena* arena;
} optimizer;

optimizer* optimizer_sgd_create(mem_arena* arena, f32 lr);
optimizer* optimizer_sgd_momentum_create(mem_arena* arena, f32 lr, f32 momentum);

// 核心更新函数：传入参数和对应梯度
void optimizer_step(optimizer* opt, matrix** params, matrix** grads, u32 num_params);

// 工具函数
void optimizer_clip_grad_norm(matrix** grads, u32 num_grads, f32 max_norm);

#endif