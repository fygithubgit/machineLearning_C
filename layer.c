#include "layer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


// 内部辅助函数声明
static matrix* layer_forward_dense(layer* l, const matrix* input, forward_ctx* ctx);
static matrix* layer_forward_activation(layer* l, const matrix* input, forward_ctx* ctx);
static matrix* layer_forward_softmax(layer* l, const matrix* input, forward_ctx* ctx);

static matrix* layer_backward_dense(layer* l, const matrix* grad_output, forward_ctx* ctx);
static matrix* layer_backward_activation(layer* l, const matrix* grad_output, forward_ctx* ctx);
static matrix* layer_backward_softmax(layer* l, const matrix* grad_output, forward_ctx* ctx);

/**
 * 创建全连接层（完整版）
 */
layer* layer_create_dense(mem_arena* arena, u32 input_size, u32 output_size, bool require_grad) {
    if (!arena || input_size == 0 || output_size == 0) {
        fprintf(stderr, "Error: Invalid parameters for dense layer\n");
        return NULL;
    }
    
    layer* l = ARENA_ALLOC_STRUCT(arena, layer);
    if (!l) {
        fprintf(stderr, "Error: Failed to allocate layer structure\n");
        return NULL;
    }
    
    // 初始化
    memset(l, 0, sizeof(layer));
    l->type = LAYER_DENSE;
    l->config.input_size = input_size;
    l->config.output_size = output_size;
    l->config.require_grad = require_grad;
    l->arena = arena;
    
    size_t temp_arena_size = 1024 * 1024;  // 1MB 临时空间足够
    // 创建临时Arena用于中间计算
    l->temp_arena = arena_create_child(arena, temp_arena_size);
    if (!l->temp_arena) {
        fprintf(stderr, "Error: Failed to create temp arena\n");
        return NULL;
    }
    
    // 创建权重 [output_size x input_size]
    mat_options weight_opts = MAT_DEFAULT_OPTS;
    weight_opts.require_grad = require_grad;
    weight_opts.zero_init = true;
    l->weights = mat_create(arena, output_size, input_size, weight_opts);
    
    // 创建偏置 [output_size x 1]  
    mat_options bias_opts = MAT_DEFAULT_OPTS;
    bias_opts.require_grad = require_grad;
    l->bias = mat_create(arena, output_size, 1, bias_opts);
    
    if (!l->weights || !l->bias) {
        fprintf(stderr, "Error: Failed to create parameters\n");
        return NULL;
    }
    
    // 创建梯度（如果需要）
    if (require_grad) {
        l->grad_weights = mat_create(arena, output_size, input_size, MAT_DEFAULT_OPTS);
        l->grad_bias = mat_create(arena, output_size, 1, MAT_DEFAULT_OPTS);
        if (!l->grad_weights || !l->grad_bias) {
            fprintf(stderr, "Error: Failed to create gradient matrices\n");
            return NULL;
        }
    }
    
    printf("Created dense layer: %u -> %u\n", input_size, output_size);
    return l;
}

/**
 * 创建激活层（完整版）
 */
layer* layer_create_activation(mem_arena* arena, activation_type act) {
    if (!arena) {
        fprintf(stderr, "Error: Invalid arena\n");
        return NULL;
    }
    
    layer* l = ARENA_ALLOC_STRUCT(arena, layer);
    if (!l) {
        fprintf(stderr, "Error: Failed to allocate activation layer\n");
        return NULL;
    }
    
    memset(l, 0, sizeof(layer));
    l->type = LAYER_ACTIVATION;
    l->config.act = act;
    l->config.require_grad = true; // 激活层总是需要梯度
    l->arena = arena;
    
    size_t temp_arena_size = 512 * 1024;  // 512KB 临时空间
    l->temp_arena = arena_create_child(arena, temp_arena_size);
    if (!l->temp_arena) {
        fprintf(stderr, "Error: Failed to create temp arena for activation\n");
        return NULL;
    }
    
    printf("Created activation layer: %d\n", act);
    return l;
}

/**
 * 创建Softmax层
 */
layer* layer_create_softmax(mem_arena* arena) {
    if (!arena) {
        fprintf(stderr, "Error: Invalid arena\n");
        return NULL;
    }
    
    layer* l = ARENA_ALLOC_STRUCT(arena, layer);
    if (!l) {
        fprintf(stderr, "Error: Failed to allocate softmax layer\n");
        return NULL;
    }
    
    memset(l, 0, sizeof(layer));
    l->type = LAYER_SOFTMAX;
    l->config.act = ACTIVATION_SOFTMAX;
    l->config.require_grad = true;
    l->arena = arena;
    
    size_t temp_arena_size = 512 * 1024;  // 512KB 临时空间
    l->temp_arena = arena_create_child(arena, temp_arena_size);
    if (!l->temp_arena) {
        fprintf(stderr, "Error: Failed to create temp arena for softmax\n");
        return NULL;
    }
    
    printf("Created softmax layer\n");
    return l;
}

/**
 * 前向传播主函数
 */
matrix* layer_forward(layer* l, const matrix* input, forward_ctx* ctx) {
    if (!l || !input) {
        fprintf(stderr, "Error: Invalid layer or input\n");
        return NULL;
    }
    
    if (!mat_is_valid(input)) {
        fprintf(stderr, "Error: Invalid input matrix\n");
        return NULL;
    }
    
    
    matrix* output = NULL;
    switch (l->type) {
        case LAYER_DENSE:
            output = layer_forward_dense(l, input, ctx);
            break;
        case LAYER_ACTIVATION:
            output = layer_forward_activation(l, input, ctx);
            break;
        case LAYER_SOFTMAX:
            output = layer_forward_softmax(l, input, ctx);
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type\n");
            return NULL;
    }
    
    // 缓存输出
    if (output && (!l->cached_output || 
                   l->cached_output->rows != output->rows || 
                   l->cached_output->cols != output->cols)) {
        mat_options opts = MAT_DEFAULT_OPTS;
        l->cached_output = mat_create(l->arena, output->rows, output->cols, opts);
    }
    if (l->cached_output) {
        mat_copy(output, l->cached_output);
        output = l->cached_output;
    }
    
    //  在forward结束后重置临时arena（此时output已复制到持久位置）
    arena_reset(l->temp_arena);
    
    return output;
}
/**
 * 全连接层前向传播（完整实现）
 * output = input * weights^T + bias
 */
// layer.c 中的 layer_forward_dense
static matrix* layer_forward_dense(layer* l, const matrix* input, forward_ctx* ctx) {
    u32 batch_size = input->rows;
    u32 output_size = l->config.output_size;
    u32 input_size = l->config.input_size;
    
    if (input->cols != input_size) {
        fprintf(stderr, "Error: Input size mismatch\n");
        return NULL;
    }
    
    // 使用临时arena进行计算
    mat_options opts = MAT_DEFAULT_OPTS;
    matrix* temp_output = mat_create(l->temp_arena, batch_size, output_size, opts);
    if (!temp_output) {
        fprintf(stderr, "Error: Failed to allocate temp output\n");
        return NULL;
    }
    
    // 执行计算：output = input * weights^T + bias
    if (mat_is_contiguous(input) && mat_is_contiguous(l->weights) && mat_is_contiguous(temp_output)) {
        for (u32 i = 0; i < batch_size; i++) {
            for (u32 j = 0; j < output_size; j++) {
                f32 sum = 0.0f;
                for (u32 k = 0; k < input_size; k++) {
                    sum += input->data[i * input_size + k] * l->weights->data[j * input_size + k];
                }
                temp_output->data[i * output_size + j] = sum;
            }
        }
    } else {
        for (u32 i = 0; i < batch_size; i++) {
            for (u32 j = 0; j < output_size; j++) {
                f32 sum = 0.0f;
                for (u32 k = 0; k < input_size; k++) {
                    sum += mat_at(input, i, k) * mat_at(l->weights, j, k);
                }
                mat_set_at(temp_output, i, j, sum);
            }
        }
    }
    
    // 广播加偏置
    mat_broadcast_add_col(temp_output, l->bias, temp_output);
    
    // 将结果复制到主arena（持久化存储）
    matrix* persistent_output = mat_create(l->arena, batch_size, output_size, opts);
    if (!persistent_output) {
        fprintf(stderr, "Error: Failed to allocate persistent output\n");
        return NULL;
    }
    
    mat_copy(temp_output, persistent_output);
    
    // 保存上下文（使用主arena的指针）
    if (ctx && l->config.require_grad) {
        ctx->input = (matrix*)input;  // 输入通常是持久的
        ctx->output = persistent_output;  // 现在是持久化的，不会被reset覆盖
        ctx->weights = l->weights;
        ctx->temp_arena = l->temp_arena;
    }
    
    return persistent_output;
}
/**
 * 激活层前向传播（完整实现）
 */
static matrix* layer_forward_activation(layer* l, const matrix* input, forward_ctx* ctx) {
    u32 batch_size = input->rows;
    u32 cols = input->cols;
    
    // 临时计算
    mat_options opts = MAT_DEFAULT_OPTS;
    matrix* temp_output = mat_create(l->temp_arena, batch_size, cols, opts);
    if (!temp_output) {
        fprintf(stderr, "Error: Failed to allocate temp activation output\n");
        return NULL;
    }
    
    // 执行激活
    switch (l->config.act) {
        case ACTIVATION_RELU:
            mat_relu(input, temp_output);
            break;
        case ACTIVATION_SIGMOID:
            mat_sigmoid(input, temp_output);
            break;
        case ACTIVATION_TANH:
            mat_tanh(input, temp_output);
            break;
        case ACTIVATION_NONE:
            mat_copy(input, temp_output);
            break;
        default:
            fprintf(stderr, "Error: Unsupported activation %d\n", l->config.act);
            return NULL;
    }
    
    // 复制到主arena
    matrix* persistent_output = mat_create(l->arena, batch_size, cols, opts);
    if (!persistent_output) return NULL;
    
    mat_copy(temp_output, persistent_output);
    
    // 保存上下文
    if (ctx && l->config.require_grad) {
        ctx->input = (matrix*)input;
        ctx->output = persistent_output;  // 持久化
        ctx->act_type = l->config.act;
        ctx->temp_arena = l->temp_arena;
    }
    
    return persistent_output;
}
/**
 * Softmax层前向传播
 */
static matrix* layer_forward_softmax(layer* l, const matrix* input, forward_ctx* ctx) {
    u32 batch_size = input->rows;
    u32 cols = input->cols;
    
    // 临时计算
    mat_options opts = MAT_DEFAULT_OPTS;
    matrix* temp_output = mat_create(l->temp_arena, batch_size, cols, opts);
    if (!temp_output) {
        fprintf(stderr, "Error: Failed to allocate temp softmax output\n");
        return NULL;
    }
    
    mat_softmax(input, temp_output);
    
    // 复制到主arena
    matrix* persistent_output = mat_create(l->arena, batch_size, cols, opts);
    if (!persistent_output) return NULL;
    
    mat_copy(temp_output, persistent_output);
    
    // 保存上下文
    if (ctx && l->config.require_grad) {
        ctx->input = (matrix*)input;
        ctx->output = persistent_output;  // 持久化
        ctx->act_type = ACTIVATION_SOFTMAX;
        ctx->temp_arena = l->temp_arena;
    }
    
    return persistent_output;
}
/**
 * 反向传播主函数
 */
matrix* layer_backward(layer* l, const matrix* grad_output, forward_ctx* ctx) {
    if (!l || !grad_output || !ctx || !l->config.require_grad) {
        fprintf(stderr, "Error: Invalid backward pass parameters\n");
        return NULL;
    }
    
    if (!mat_is_valid(grad_output)) {
        fprintf(stderr, "Error: Invalid grad_output matrix\n");
        return NULL;
    }
    
    
    matrix* grad_input = NULL;
    switch (l->type) {
        case LAYER_DENSE:
            grad_input = layer_backward_dense(l, grad_output, ctx);
            break;
        case LAYER_ACTIVATION:
            grad_input = layer_backward_activation(l, grad_output, ctx);
            break;
        case LAYER_SOFTMAX:
            grad_input = layer_backward_softmax(l, grad_output, ctx);
            break;
        default:
            fprintf(stderr, "Error: Unknown layer type in backward\n");
            return NULL;
    }
    
    // ✅ 关键修复：先检查和缓存 grad_input，再重置 arena
    if (grad_input) {
        // 1. 先创建持久化的缓存矩阵（在主arena中）
        if (!l->cached_grad_input || 
            l->cached_grad_input->rows != grad_input->rows || 
            l->cached_grad_input->cols != grad_input->cols) {
            
            mat_options opts = MAT_DEFAULT_OPTS;  // 或适当的选项
            l->cached_grad_input = mat_create(l->arena, grad_input->rows, grad_input->cols, opts);
        }
        
        // 2. 将临时结果复制到持久缓存中（在重置arena之前）
        if (l->cached_grad_input) {
            mat_copy(grad_input, l->cached_grad_input);
        }
    }
    
    // 3. ✅ 现在安全了，可以重置临时arena（grad_input 已被复制）
    arena_reset(l->temp_arena);
    
    // 4. 返回持久化的缓存结果
    return l->cached_grad_input ? l->cached_grad_input : grad_input;
}
/**
 * 全连接层反向传播（完整实现）
 * 
 * 给定: grad_output [batch_size x output_size]
 * 计算: 
 *   grad_weights = grad_output^T * input
 *   grad_bias = sum(grad_output, axis=0)
 *   grad_input = grad_output * weights
 */
static matrix* layer_backward_dense(layer* l, const matrix* grad_output, forward_ctx* ctx) {
    if (!grad_output || !ctx) return NULL;
    
    matrix* input = ctx->input;
    matrix* output = ctx->output;  // 现在是持久化的，不会被reset覆盖
    matrix* weights = ctx->weights;
    
    if (!input || !output || !weights) {
        fprintf(stderr, "Error: Missing context data in backward\n");
        return NULL;
    }
    
    u32 batch_size = grad_output->rows;
    u32 input_size = input->cols;
    u32 output_size = grad_output->cols;
    
    // 1. 计算权重梯度（使用临时arena）: temp_grad_weights [output_size x input_size]
    matrix* temp_grad_weights = mat_create(l->temp_arena, output_size, input_size, MAT_DEFAULT_OPTS);
    if (!temp_grad_weights) return NULL;

    // 2. 计算偏置梯度（使用临时arena）: bias is [output_size x 1]
    matrix* temp_grad_bias = mat_create(l->temp_arena, output_size, 1, MAT_DEFAULT_OPTS);
    if (!temp_grad_bias) return NULL;

    // 3. 计算输入梯度（使用临时arena）: temp_grad_input [batch_size x input_size]
    matrix* temp_grad_input = mat_create(l->temp_arena, batch_size, input_size, MAT_DEFAULT_OPTS);
    if (!temp_grad_input) return NULL;

    // Compute gradients
    // grad_weights[j,k] = sum_i grad_output[i,j] * input[i,k]
    if (mat_is_contiguous(grad_output) && mat_is_contiguous(input) && mat_is_contiguous(temp_grad_weights)) {
        for (u32 j = 0; j < output_size; j++) {
            for (u32 k = 0; k < input_size; k++) {
                f32 sum = 0.0f;
                for (u32 i = 0; i < batch_size; i++) {
                    sum += grad_output->data[i * output_size + j] * input->data[i * input_size + k];
                }
                temp_grad_weights->data[j * input_size + k] = sum;
            }
        }
    } else {
        for (u32 j = 0; j < output_size; j++) {
            for (u32 k = 0; k < input_size; k++) {
                f32 sum = 0.0f;
                for (u32 i = 0; i < batch_size; i++) {
                    sum += mat_at(grad_output, i, j) * mat_at(input, i, k);
                }
                mat_set_at(temp_grad_weights, j, k, sum);
            }
        }
    }

    // grad_bias[j,0] = sum_i grad_output[i,j]
    if (mat_is_contiguous(grad_output) && mat_is_contiguous(temp_grad_bias)) {
        // initialize bias gradient to 0
        for (u32 j = 0; j < output_size; j++) temp_grad_bias->data[j] = 0.0f;
        for (u32 i = 0; i < batch_size; i++) {
            for (u32 j = 0; j < output_size; j++) {
                temp_grad_bias->data[j] += grad_output->data[i * output_size + j];
            }
        }
    } else {
        for (u32 j = 0; j < output_size; j++) mat_set_at(temp_grad_bias, j, 0, 0.0f);
        for (u32 i = 0; i < batch_size; i++) {
            for (u32 j = 0; j < output_size; j++) {
                f32 v = mat_at(temp_grad_bias, j, 0) + mat_at(grad_output, i, j);
                mat_set_at(temp_grad_bias, j, 0, v);
            }
        }
    }

    // grad_input[i,k] = sum_j grad_output[i,j] * weights[j,k]
    if (mat_is_contiguous(grad_output) && mat_is_contiguous(weights) && mat_is_contiguous(temp_grad_input)) {
        for (u32 i = 0; i < batch_size; i++) {
            for (u32 k = 0; k < input_size; k++) {
                f32 sum = 0.0f;
                for (u32 j = 0; j < output_size; j++) {
                    sum += grad_output->data[i * output_size + j] * weights->data[j * input_size + k];
                }
                temp_grad_input->data[i * input_size + k] = sum;
            }
        }
    } else {
        for (u32 i = 0; i < batch_size; i++) {
            for (u32 k = 0; k < input_size; k++) {
                f32 sum = 0.0f;
                for (u32 j = 0; j < output_size; j++) {
                    sum += mat_at(grad_output, i, j) * mat_at(weights, j, k);
                }
                mat_set_at(temp_grad_input, i, k, sum);
            }
        }
    }
    
    // 4. 将临时结果复制到主arena
    matrix* final_grad_weights = mat_create(l->arena, temp_grad_weights->rows, temp_grad_weights->cols, MAT_DEFAULT_OPTS);
    matrix* final_grad_bias = mat_create(l->arena, temp_grad_bias->rows, temp_grad_bias->cols, MAT_DEFAULT_OPTS);
    matrix* final_grad_input = mat_create(l->arena, temp_grad_input->rows, temp_grad_input->cols, MAT_DEFAULT_OPTS);
    
    if (!final_grad_weights || !final_grad_bias || !final_grad_input) {
        fprintf(stderr, "Error: Failed to create persistent gradients\n");
        return NULL;
    }
    
    mat_copy(temp_grad_weights, final_grad_weights);
    mat_copy(temp_grad_bias, final_grad_bias);
    mat_copy(temp_grad_input, final_grad_input);
    
    // 5. 更新层的梯度（现在是持久化的）
    if (l->grad_weights) mat_add_inplace(l->grad_weights, final_grad_weights);
    else l->grad_weights = final_grad_weights;
    
    if (l->grad_bias) mat_add_inplace(l->grad_bias, final_grad_bias);
    else l->grad_bias = final_grad_bias;
    
    // 6. 返回输入梯度（也是持久化的）
    return final_grad_input;
}
/**
 * ReLU反向传播
 * grad_input = grad_output * (input > 0)
 */
static void relu_backward(const matrix* input, const matrix* grad_output, matrix* grad_input) {
    u32 rows = input->rows;
    u32 cols = input->cols;
    
    if (mat_is_contiguous(input) && mat_is_contiguous(grad_output) && mat_is_contiguous(grad_input)) {
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            grad_input->data[i] = (input->data[i] > 0.0f) ? grad_output->data[i] : 0.0f;
        }
    } else {
        for (u32 i = 0; i < rows; i++) {
            for (u32 j = 0; j < cols; j++) {
                f32 input_val = mat_at(input, i, j);
                f32 grad_out_val = mat_at(grad_output, i, j);
                mat_set_at(grad_input, i, j, (input_val > 0.0f) ? grad_out_val : 0.0f);
            }
        }
    }
}

/**
 * Sigmoid反向传播
 * grad_input = grad_output * output * (1 - output)
 */
static void sigmoid_backward(const matrix* output, const matrix* grad_output, matrix* grad_input) {
    u32 rows = output->rows;
    u32 cols = output->cols;
    
    if (mat_is_contiguous(output) && mat_is_contiguous(grad_output) && mat_is_contiguous(grad_input)) {
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            f32 out_val = output->data[i];
            grad_input->data[i] = grad_output->data[i] * out_val * (1.0f - out_val);
        }
    } else {
        for (u32 i = 0; i < rows; i++) {
            for (u32 j = 0; j < cols; j++) {
                f32 out_val = mat_at(output, i, j);
                f32 grad_out_val = mat_at(grad_output, i, j);
                mat_set_at(grad_input, i, j, grad_out_val * out_val * (1.0f - out_val));
            }
        }
    }
}

/**
 * Tanh反向传播  
 * grad_input = grad_output * (1 - output^2)
 */
static void tanh_backward(const matrix* output, const matrix* grad_output, matrix* grad_input) {
    u32 rows = output->rows;
    u32 cols = output->cols;
    
    if (mat_is_contiguous(output) && mat_is_contiguous(grad_output) && mat_is_contiguous(grad_input)) {
        for (size_t i = 0; i < (size_t)rows * cols; i++) {
            f32 out_val = output->data[i];
            grad_input->data[i] = grad_output->data[i] * (1.0f - out_val * out_val);
        }
    } else {
        for (u32 i = 0; i < rows; i++) {
            for (u32 j = 0; j < cols; j++) {
                f32 out_val = mat_at(output, i, j);
                f32 grad_out_val = mat_at(grad_output, i, j);
                mat_set_at(grad_input, i, j, grad_out_val * (1.0f - out_val * out_val));
            }
        }
    }
}

/**
 * Softmax反向传播
 * 对于交叉熵损失，softmax + cross-entropy 的组合梯度为: output - target
 * 这里实现通用的softmax反向传播:
 * grad_input = output * (grad_output - sum(output * grad_output))
 */
static void softmax_backward(const matrix* output, const matrix* grad_output, matrix* grad_input) {
    u32 batch_size = output->rows;
    u32 num_classes = output->cols;
    
    if (!output || !grad_output || !grad_input || 
        output->rows != grad_output->rows || output->cols != grad_output->cols ||
        output->rows != grad_input->rows || output->cols != grad_input->cols) {
        fprintf(stderr, "Error: Invalid dimensions in softmax backward\n");
        return;
    }
    
    //  创建正确维度的临时矩阵
    matrix* temp = mat_create(grad_input->arena, batch_size, num_classes, MAT_DEFAULT_OPTS);
    if (!temp) {
        fprintf(stderr, "Error: Failed to allocate temp matrix in softmax backward\n");
        return;
    }
    
    //  创建正确维度的sum_buffer [batch_size x 1] - 用于行求和
    matrix* sum_buffer = mat_create(grad_input->arena, batch_size, 1, MAT_DEFAULT_OPTS);
    if (!sum_buffer) {
        fprintf(stderr, "Error: Failed to allocate sum buffer in softmax backward\n");
        return;
    }
    
    // 步骤1：计算 output * grad_output
    mat_elementwise_mul(output, grad_output, temp);
    
    // 对每个样本的所有类别求和：sum_j(output_ij * grad_output_ij)
    mat_sum_rows(temp, sum_buffer);  // sum_buffer[i][0] = sum over j of (temp[i][j])
    
    // 步骤2：广播减法，将 [batch_size x 1] 的 sum_buffer 广播到 [batch_size x num_classes]
    // grad_output - sum_buffer (广播)
    for (u32 i = 0; i < batch_size; i++) {
        f32 row_sum = mat_at(sum_buffer, i, 0);  // sum_j(output_ij * grad_output_ij)
        for (u32 j = 0; j < num_classes; j++) {
            f32 grad_out_val = mat_at(grad_output, i, j);
            mat_set_at(temp, i, j, grad_out_val - row_sum);  // grad_output_ij - sum_term
        }
    }
    
    // 步骤3：逐元素乘以 output
    // grad_input = output * (grad_output - sum_over_classes(output * grad_output))
    mat_elementwise_mul_inplace(temp, output);
    
    // 步骤4：复制结果
    mat_copy(temp, grad_input);
}
/**
 * 激活层反向传播
 */
static matrix* layer_backward_activation(layer* l, const matrix* grad_output, forward_ctx* ctx) {
    if (!ctx->input || !ctx->output) {
        fprintf(stderr, "Error: Missing context in activation backward\n");
        return NULL;
    }
    
    u32 batch_size = grad_output->rows;
    u32 cols = grad_output->cols;
    
    matrix* grad_input = mat_create(l->temp_arena, batch_size, cols, MAT_DEFAULT_OPTS);
    if (!grad_input) {
        fprintf(stderr, "Error: Failed to allocate activation grad_input\n");
        return NULL;
    }
    
    switch (ctx->act_type) {
        case ACTIVATION_RELU:
            relu_backward(ctx->input, grad_output, grad_input);
            break;
        case ACTIVATION_SIGMOID:
            sigmoid_backward(ctx->output, grad_output, grad_input);
            break;
        case ACTIVATION_TANH:
            tanh_backward(ctx->output, grad_output, grad_input);
            break;
        case ACTIVATION_SOFTMAX:
            softmax_backward(ctx->output, grad_output, grad_input);
            break;
        case ACTIVATION_NONE:
            mat_copy(grad_output, grad_input);
            break;
        default:
            fprintf(stderr, "Error: Unsupported activation in backward %d\n", ctx->act_type);
            return NULL;
    }
    
    return grad_input;
}

/**
 * Softmax层反向传播
 */
static matrix* layer_backward_softmax(layer* l, const matrix* grad_output, forward_ctx* ctx) {
    return layer_backward_activation(l, grad_output, ctx);
}

/**
 * Xavier初始化
 */
void layer_init_xavier(layer* l, u64 seed) {
    if (!l || l->type != LAYER_DENSE) return;
    
    f32 limit = sqrtf(6.0f / (f32)(l->config.input_size + l->config.output_size));
    mat_random_uniform(l->weights, -limit, limit, &seed);
    mat_fill(l->bias, 0.0f);
}

/**
 * He初始化  
 */
void layer_init_he(layer* l, u64 seed) {
    if (!l || l->type != LAYER_DENSE) return;
    
    f32 stddev = sqrtf(2.0f / (f32)l->config.input_size);
    mat_random_normal(l->weights, 0.0f, stddev, &seed);
    mat_fill(l->bias, 0.0f);
}

/**
 * 零初始化
 */
void layer_init_zeros(layer* l) {
    if (!l || l->type != LAYER_DENSE) return;
    mat_fill(l->weights, 0.0f);
    mat_fill(l->bias, 0.0f);
}

/**
 * 清零梯度
 */
void layer_zero_grad(layer* l) {
    if (!l || !l->config.require_grad) return;
    
    if (l->grad_weights) mat_fill(l->grad_weights, 0.0f);
    if (l->grad_bias) mat_fill(l->grad_bias, 0.0f);
}

// 其他辅助函数...

/**
 * 创建前向传播上下文
 */
forward_ctx* forward_ctx_create(mem_arena* arena) {
    forward_ctx* ctx = ARENA_ALLOC_STRUCT(arena, forward_ctx);
    if (ctx) {
        memset(ctx, 0, sizeof(forward_ctx));
    }
    return ctx;
}

/**
 * 重置前向传播上下文
 */
void forward_ctx_reset(forward_ctx* ctx) {
    if (ctx) {
        memset(ctx, 0, sizeof(forward_ctx));
    }
}

// 参数访问函数
matrix* layer_get_weights(const layer* l) { return l ? l->weights : NULL; }
matrix* layer_get_bias(const layer* l) { return l ? l->bias : NULL; }
matrix* layer_get_grad_weights(const layer* l) { return l ? l->grad_weights : NULL; }
matrix* layer_get_grad_bias(const layer* l) { return l ? l->grad_bias : NULL; }

void layer_print_info(const layer* l, const char* name) {
    if (!l) return;
    
    const char* type_names[] = {"DENSE", "ACTIVATION", "SOFTMAX"};
    const char* act_names[] = {"NONE", "RELU", "SIGMOID", "TANH", "SOFTMAX"};
    
    printf("%s Layer Info:\n", name ? name : "Layer");
    printf("  Type: %s\n", type_names[l->type]);
    printf("  Input Size: %u\n", l->config.input_size);
    printf("  Output Size: %u\n", l->config.output_size);
    printf("  Activation: %s\n", act_names[l->config.act]);
    printf("  Requires Grad: %s\n", l->config.require_grad ? "true" : "false");
    
    if (l->type == LAYER_DENSE) {
        printf("  Parameters: %u\n", layer_get_param_count(l));
    }
    printf("\n");
}

u32 layer_get_param_count(const layer* l) {
    if (!l || l->type != LAYER_DENSE) return 0;
    return l->weights->rows * l->weights->cols + l->bias->rows;
}

f32 layer_get_total_norm(const layer* l) {
    if (!l || l->type != LAYER_DENSE) return 0.0f;
    f32 weight_norm_sq = 0.0f, bias_norm_sq = 0.0f;
    
    // 计算权重L2范数平方
    if (mat_is_contiguous(l->weights)) {
        size_t elements = mat_size(l->weights);
        for (size_t i = 0; i < elements; i++) {
            weight_norm_sq += l->weights->data[i] * l->weights->data[i];
        }
    }
    
    // 计算偏置L2范数平方  
    if (mat_is_contiguous(l->bias)) {
        size_t elements = mat_size(l->bias);
        for (size_t i = 0; i < elements; i++) {
            bias_norm_sq += l->bias->data[i] * l->bias->data[i];
        }
    }
    
    return sqrtf(weight_norm_sq + bias_norm_sq);
}