#ifndef MATRIX_H
#define MATRIX_H

#include "arena.h"
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

typedef uint32_t u32;
typedef float f32;

// 矩阵标志位
#define MAT_FLAG_CONTIGUOUS (1 << 0)
#define MAT_FLAG_OWNED      (1 << 1)

// 矩阵创建选项
typedef struct {
    bool require_grad;  // 是否需要梯度（预留）
    bool zero_init;     // 是否零初始化
    f32 init_value;     // 初始化值（当zero_init=false时使用）
} mat_options;

// 矩阵结构体
typedef struct {
    u32 rows;
    u32 cols;
    u32 row_stride;     // 行步长（元素数）
    u32 col_stride;     // 列步长（元素数）  
    f32* data;          // 数据指针
    u32 flags;          // 标志位
    mem_arena* arena;   // 关联的内存分配器（仅当OWNED时有效）
    mem_arena* temp_arena;  // 临时内存分配器（用于临时计算）
} matrix;

// 默认选项
#define MAT_DEFAULT_OPTS ((mat_options){ .require_grad = false, .zero_init = true, .init_value = 0.0f })

// === 基础操作 ===
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols, mat_options opts);
void mat_destroy(matrix* m);  // 实际由Arena管理
bool mat_is_valid(const matrix* m);
bool mat_is_contiguous(const matrix* m);
size_t mat_size(const matrix* m);
f32 mat_at(const matrix* m, u32 row, u32 col);
void mat_set_at(matrix* m, u32 row, u32 col, f32 value);
void mat_fill(matrix* m, f32 value);
void mat_copy(const matrix* src, matrix* dst);

// === 矩阵运算 ===
void mat_add(const matrix* a, const matrix* b, matrix* result);
void mat_sub(const matrix* a, const matrix* b, matrix* result);
void mat_mul_scalar(const matrix* a, f32 scalar, matrix* result);
void mat_matmul(const matrix* a, const matrix* b, matrix* result);

// === 统计函数 ===
f32 mat_sum(const matrix* m);
f32 mat_sum_of_squares(const matrix* m);
f32 mat_mean(const matrix* m);
f32 mat_variance(const matrix* m);

// === 聚合操作 ===
void mat_sum_rows(const matrix* m, matrix* result);  // [rows x cols] -> [rows x 1]
void mat_sum_cols(const matrix* m, matrix* result);  // [rows x cols] -> [1 x cols]

// === 广播操作 ===
void mat_broadcast_add_col(const matrix* m, const matrix* col_vec, matrix* result); // m + col_vec
void mat_broadcast_sub_col(const matrix* m, const matrix* col_vec, matrix* result); // m - col_vec
void mat_broadcast_add_row(const matrix* m, const matrix* row_vec, matrix* result); // m + row_vec
void mat_broadcast_sub_row(const matrix* m, const matrix* row_vec, matrix* result); // m - row_vec

// === 元素级操作 ===
void mat_elementwise_mul(const matrix* a, const matrix* b, matrix* result);
void mat_elementwise_mul_inplace(matrix* a, const matrix* b); // a *= b
void mat_elementwise_div(const matrix* a, const matrix* b, matrix* result);
void mat_elementwise_div_inplace(matrix* a, const matrix* b); // a /= b

// 额外实用函数
void mat_scale_inplace(matrix* m, f32 scalar);
void mat_add_inplace(matrix* a, const matrix* b);
void mat_sub_inplace(matrix* a, const matrix* b);
void mat_axpy_inplace(matrix* y, f32 alpha, const matrix* x);  // y = y + alpha*x

// === 转置操作 ===
void mat_transpose(const matrix* m, matrix* result);

// === 随机数生成 ===
void mat_random_uniform(matrix* m, f32 min_val, f32 max_val, u64* seed);
void mat_random_normal(matrix* m, f32 mean, f32 stddev, u64* seed);

// === 激活函数 ===
void mat_relu(const matrix* input, matrix* output);
void mat_sigmoid(const matrix* input, matrix* output);
void mat_tanh(const matrix* input, matrix* output);
void mat_softmax(const matrix* input, matrix* output);

// === 数学常量 ===
#define MATH_PI_F 3.14159265358979323846f
#define MATH_E_F  2.71828182845904523536f

#endif // MATRIX_H
