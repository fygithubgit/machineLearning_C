#include "model.h"
#include <stdio.h>
#include <string.h>

static void collect_layer_params(layer* l, matrix*** param_list, u32* count, u32* capacity, mem_arena* arena) {
    if (!l) return;
    matrix* w = layer_get_weights(l);
    matrix* b = layer_get_bias(l);
    
    if (w) {
        if (*count >= *capacity) {
            *capacity = (*capacity == 0) ? 4 : (*capacity * 2);
            matrix** new_list = (matrix**)mem_arena_alloc(arena, sizeof(matrix*) * (*capacity), ARENA_DEFAULT_ALIGNMENT);
            if (*param_list) memcpy(new_list, *param_list, sizeof(matrix*) * (*count));
            *param_list = new_list;
        }
        (*param_list)[(*count)++] = w;
    }
    if (b) {
        if (*count >= *capacity) {
            *capacity = (*capacity == 0) ? 4 : (*capacity * 2);
            matrix** new_list = (matrix**)mem_arena_alloc(arena, sizeof(matrix*) * (*capacity), ARENA_DEFAULT_ALIGNMENT);
            if (*param_list) memcpy(new_list, *param_list, sizeof(matrix*) * (*count));
            *param_list = new_list;
        }
        (*param_list)[(*count)++] = b;
    }
}

model* model_create(mem_arena* arena, u32 max_layers) {
    if (!arena || max_layers == 0) return NULL;
    
    model* m = ARENA_ALLOC_STRUCT(arena, model);
    if (!m) return NULL;
    
    m->layers = (layer**)mem_arena_alloc(
        arena, 
        sizeof(layer*) * max_layers, 
        ARENA_DEFAULT_ALIGNMENT
    );
    m->forward_contexts = (forward_ctx**)mem_arena_alloc(
        arena, 
        sizeof(forward_ctx*) * max_layers, 
        ARENA_DEFAULT_ALIGNMENT
    );
    
    if (!m->layers || !m->forward_contexts) return NULL;
    
    m->max_layers = max_layers;
    m->num_layers = 0;
    m->arena = arena;
    
    return m;
}

bool model_add_layer(model* m, layer* l) {
    if (!m || !l || m->num_layers >= m->max_layers) return false;
    m->layers[m->num_layers] = l;
    m->num_layers++;
    return true;
}

matrix* model_forward(model* m, const matrix* input) {
    if (!m || !input) return NULL;
    matrix* current = (matrix*)input;
    for (u32 i = 0; i < m->num_layers; i++) {
        if (!m->forward_contexts[i]) {
            m->forward_contexts[i] = forward_ctx_create(m->arena);
        }
        current = layer_forward(m->layers[i], current, m->forward_contexts[i]);
        if (!current) {
            fprintf(stderr, "Forward failed at layer %u\n", i);
            return NULL;
        }
    }
    return current;
}

matrix** model_get_parameters(model* m, u32* out_count) {
    if (!m) { if (out_count) *out_count = 0; return NULL; }
    if (m->parameters) {
        if (out_count) *out_count = m->num_parameters;
        return m->parameters;
    }
    
    u32 capacity = 8, count = 0;
    matrix** params = (matrix**)mem_arena_alloc(m->arena, sizeof(matrix*) * capacity, ARENA_DEFAULT_ALIGNMENT);
    for (u32 i = 0; i < m->num_layers; i++) {
        collect_layer_params(m->layers[i], &params, &count, &capacity, m->arena);
    }
    
    m->parameters = params;
    m->num_parameters = count;
    if (out_count) *out_count = count;
    return params;
}

void model_summary(const model* m) {
    if (!m) { printf("Model: NULL\n"); return; }
    printf("=== Model Summary ===\n");
    size_t total = 0;
    for (u32 i = 0; i < m->num_layers; i++) {
        layer* l = m->layers[i];
        if (!l) continue;
        switch (l->type) {
            case LAYER_LINEAR: {
                size_t p = mat_size(layer_get_weights(l)) + mat_size(layer_get_bias(l));
                total += p;
                printf("Layer %u: Linear (%zu params)\n", i, p);
                break;
            }
            case LAYER_ACTIVATION:
                printf("Layer %u: Activation\n", i);
                break;
            default:
                printf("Layer %u: Unknown\n", i);
        }
    }
    printf("Total parameters: %zu\n", total);
    printf("=====================\n");
}

matrix* model_backward(model* m, const matrix* grad_output) {
    if (!m || !grad_output) {
        fprintf(stderr, "Error: model_backward: invalid args\n");
        return NULL;
    }

    if (m->num_layers == 0) {
        fprintf(stderr, "Warning: model has no layers\n");
        return NULL;
    }

    // 检查是否已执行前向传播（即 forward_contexts 是否有效）
    for (u32 i = 0; i < m->num_layers; i++) {
        if (!m->forward_contexts[i]) {
            fprintf(stderr, "Error: Forward context missing for layer %u. Call model_forward first!\n", i);
            return NULL;
        }
    }

    // 从最后一层开始反向
    matrix* current_grad = (matrix*)grad_output;

    for (i32 i = (i32)(m->num_layers - 1); i >= 0; i--) {
        layer* l = m->layers[i];
        forward_ctx* ctx = m->forward_contexts[i];

        current_grad = layer_backward(l, current_grad, ctx);
        if (!current_grad) {
            fprintf(stderr, "Backward failed at layer %d\n", (int)i);
            return NULL;
        }
    }

    // 在model_backward结束时统一重置所有临时arena
    // 这样确保了每层的grad_input在传递给上一层时都是有效的
    for (u32 i = 0; i < m->num_layers; i++) {
        if (m->layers[i]->temp_arena) {
            arena_reset(m->layers[i]->temp_arena);
        }
    }

    return current_grad; // 即 dL/dX（输入梯度）
}
void model_zero_grad(model* m) {
    if (!m) return;
    for (u32 i = 0; i < m->num_layers; i++) {
        layer_zero_grad(m->layers[i]);
    }
}

matrix** model_get_gradients(model* m, u32* out_count) {
    if (!m) { if (out_count) *out_count = 0; return NULL; }
    
    u32 count = 0;
    u32 capacity = 8;
    matrix** grads = (matrix**)mem_arena_alloc(m->arena, sizeof(matrix*) * capacity, ARENA_DEFAULT_ALIGNMENT);
    
    for (u32 i = 0; i < m->num_layers; i++) {
        layer* l = m->layers[i];
        matrix* gw = layer_get_grad_weights(l);
        matrix* gb = layer_get_grad_bias(l);
        
        if (gw) {
            if (count >= capacity) {
                capacity *= 2;
                matrix** new_grads = (matrix**)mem_arena_alloc(m->arena, sizeof(matrix*) * capacity, ARENA_DEFAULT_ALIGNMENT);
                memcpy(new_grads, grads, sizeof(matrix*) * count);
                grads = new_grads;
            }
            grads[count++] = gw;
        }
        if (gb) {
            if (count >= capacity) {
                capacity *= 2;
                matrix** new_grads = (matrix**)mem_arena_alloc(m->arena, sizeof(matrix*) * capacity, ARENA_DEFAULT_ALIGNMENT);
                memcpy(new_grads, grads, sizeof(matrix*) * count);
                grads = new_grads;
            }
            grads[count++] = gb;
        }
    }
    
    if (out_count) *out_count = count;
    return grads;
}