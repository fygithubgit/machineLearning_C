#ifndef MODEL_H
#define MODEL_H

#include "layer.h"   // 提供 layer, layer_type, LAYER_LINEAR 等
#include "matrix.h"  // 提供 matrix, mem_arena, mat_size 等
#include <stdint.h>
#include <stdbool.h>

typedef uint32_t u32;

typedef struct model {
    u32 max_layers;
    u32 num_layers;
    layer** layers;
    forward_ctx** forward_contexts;  // 每层的前向上下文（用于反向）
    matrix** parameters;
    u32 num_parameters;
    mem_arena* arena;
} model;

// 创建模型（指定最大层数）
model* model_create(mem_arena* arena, u32 max_layers);

// 添加层
bool model_add_layer(model* m, layer* l);

// 前向传播
matrix* model_forward(model* m, const matrix* input);

matrix* model_backward(model* m, const matrix* grad_output); 

// 获取所有可训练参数
matrix** model_get_parameters(model* m, u32* out_count);

// 打印摘要
void model_summary(const model* m);

void model_zero_grad(model* m);

matrix** model_get_gradients(model* m, u32* out_count);

#endif // MODEL_H