#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include <stdbool.h>

// 前向声明
typedef struct layer layer;
typedef struct forward_ctx forward_ctx;

// 激活函数类型
typedef enum {
    ACTIVATION_NONE = 0,
    ACTIVATION_RELU,
    ACTIVATION_SIGMOID, 
    ACTIVATION_TANH,
    ACTIVATION_SOFTMAX
} activation_type;

// 层类型
typedef enum {
    LAYER_DENSE = 0,
    LAYER_ACTIVATION,
    LAYER_SOFTMAX,
    LAYER_LINEAR
} layer_type;


// 前向传播上下文 - 保存反向传播所需的所有中间结果
struct forward_ctx {
    matrix* input;          // 输入的引用（不拥有内存）
    matrix* output;         // 输出的引用（不拥有内存）
    
    // Dense层特有
    matrix* weights;        // 权重引用
    
    // 激活层特有  
    activation_type act_type;
    
    // Softmax层特有
    // (softmax的反向传播需要前向输出)
    
    mem_arena* temp_arena;  // 临时内存分配器（用于临时计算）
};

// 层配置选项
typedef struct {
    u32 input_size;         // 输入维度
    u32 output_size;        // 输出维度（仅Dense层需要）
    activation_type act;    // 激活函数类型
    bool require_grad;      // 是否需要梯度
    f32 dropout_rate;       // Dropout率（预留扩展）
} layer_config;

// 层结构体
struct layer {
    layer_type type;
    layer_config config;
    
    // 参数（仅Dense层使用）
    matrix* weights;        // 权重矩阵 [output_size x input_size]
    matrix* bias;           // 偏置向量 [output_size x 1]
    
    // 梯度（训练时使用）
    matrix* grad_weights;   // 权重梯度
    matrix* grad_bias;      // 偏置梯度
    
    // 缓存（避免重复分配）
    matrix* cached_output;  // 前向传播输出缓存
    matrix* cached_grad_input; // 反向传播输入梯度缓存
    
    mem_arena* arena;       // 主内存分配器
    mem_arena* temp_arena;  // 临时内存分配器（用于中间计算）
};

// 函数声明

// === 层创建与销毁 ===
layer* layer_create_dense(mem_arena* arena, u32 input_size, u32 output_size, bool require_grad);
layer* layer_create_activation(mem_arena* arena, activation_type act);
layer* layer_create_softmax(mem_arena* arena);

void layer_destroy(layer* l);  // 清理状态

// === 前向传播 ===
matrix* layer_forward(layer* l, const matrix* input, forward_ctx* ctx);

// === 反向传播 ===  
matrix* layer_backward(layer* l, const matrix* grad_output, forward_ctx* ctx);

// === 参数初始化 ===
void layer_init_xavier(layer* l, u64 seed);
void layer_init_he(layer* l, u64 seed);
void layer_init_zeros(layer* l);

// === 参数访问 ===
matrix* layer_get_weights(const layer* l);
matrix* layer_get_bias(const layer* l);
matrix* layer_get_grad_weights(const layer* l);
matrix* layer_get_grad_bias(const layer* l);

// === 实用函数 ===
void layer_print_info(const layer* l, const char* name);
u32 layer_get_param_count(const layer* l);
f32 layer_get_total_norm(const layer* l);
void layer_zero_grad(layer* l);  // 清零梯度

// === 辅助函数 ===
forward_ctx* forward_ctx_create(mem_arena* arena);
void forward_ctx_reset(forward_ctx* ctx);


// === 新增矩阵操作函数（在Matrix.h中也需要添加）===
void mat_sum_rows(const matrix* m, matrix* result);  // 按行求和 -> [rows x 1]
void mat_sum_cols(const matrix* m, matrix* result);  // 按列求和 -> [1 x cols]  
void mat_broadcast_add_col(const matrix* m, const matrix* col_vec, matrix* result); // 列向量广播加法
void mat_broadcast_sub_col(const matrix* m, const matrix* col_vec, matrix* result); // 列向量广播减法
void mat_elementwise_mul_inplace(matrix* a, const matrix* b); // 原地逐元素乘法
void mat_transpose(const matrix* m, matrix* result); // 转置



#endif // LAYER_H