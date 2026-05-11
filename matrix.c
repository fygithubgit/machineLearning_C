#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// 内部辅助函数
static inline f32 fast_exp(f32 x) {
    // 快速指数函数近似（可选优化）
    return expf(x);
}

static inline f32 fast_tanh(f32 x) {
    // 快速tanh近似（可选优化）
    return tanhf(x);
}

/**
 * 创建矩阵
 */
matrix* mat_create(mem_arena* arena, u32 rows, u32 cols, mat_options opts) {
    if (!arena || rows == 0 || cols == 0) {
        fprintf(stderr, "Error: Invalid matrix dimensions\n");
        return NULL;
    }
    
    // 使用新的宏
    matrix* m = ARENA_ALLOC_STRUCT(arena, matrix);
    if (!m) return NULL;
    
    // 分配数据内存
    size_t data_size = (size_t)rows * cols * sizeof(f32);
    f32* data = (f32*)mem_arena_alloc(arena, data_size, ARENA_DEFAULT_ALIGNMENT);
    if (!data) {
        fprintf(stderr, "Error: Failed to allocate matrix data\n");
        return NULL;
    }
    
    // 初始化矩阵
    m->rows = rows;
    m->cols = cols;
    m->data = data;
    //m->options = opts;
    
    m->row_stride = cols; // 对于连续内存，行步长等于列数
    m->col_stride = 1;    // 列步长为1
    m->flags = MAT_FLAG_CONTIGUOUS; //
    m->arena = arena;
    m->temp_arena = NULL;
    if (opts.zero_init) {
        size_t elements = (size_t)rows * cols;
        for (size_t i = 0; i < elements; i++) m->data[i] = 0.0f;
    }
    return m;
}

void mat_destroy(matrix* m) {
    if (!m) return;
    /* Memory is managed by arena; just clear fields to avoid dangling use */
    m->data = NULL;
    m->rows = 0;
    m->cols = 0;
    m->row_stride = 0;
    m->col_stride = 0;
    m->flags = 0;
    m->arena = NULL;
    m->temp_arena = NULL;
}

/**
 * 验证矩阵有效性
 */
bool mat_is_valid(const matrix* m) {
    return m != NULL && m->data != NULL && m->rows > 0 && m->cols > 0;
}

/**
 * 检查矩阵是否连续存储
 */
bool mat_is_contiguous(const matrix* m) {
    return m && (m->flags & MAT_FLAG_CONTIGUOUS);
}

/**
 * 获取矩阵元素总数
 */
size_t mat_size(const matrix* m) {
    return m ? (size_t)m->rows * (size_t)m->cols : 0;
}

/**
 * 获取矩阵元素（安全访问）
 */
f32 mat_at(const matrix* m, u32 row, u32 col) {
    if (!m || row >= m->rows || col >= m->cols) {
        fprintf(stderr, "Error: Matrix access out of bounds\n");
        return 0.0f;
    }
    return m->data[row * m->row_stride + col * m->col_stride];
}

/**
 * 设置矩阵元素（安全访问）
 */
void mat_set_at(matrix* m, u32 row, u32 col, f32 value) {
    if (!m || row >= m->rows || col >= m->cols) {
        fprintf(stderr, "Error: Matrix set out of bounds\n");
        return;
    }
    m->data[row * m->row_stride + col * m->col_stride] = value;
}

/**
 * 填充矩阵
 */
void mat_fill(matrix* m, f32 value) {
    if (!mat_is_valid(m)) return;
    
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            m->data[i] = value;
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                mat_set_at(m, i, j, value);
            }
        }
    }
}

/**
 * 复制矩阵
 */
void mat_copy(const matrix* src, matrix* dst) {
    if (!mat_is_valid(src) || !mat_is_valid(dst)) {
        fprintf(stderr, "Error: Invalid matrices in copy\n");
        return;
    }
    
    if (src->rows != dst->rows || src->cols != dst->cols) {
        fprintf(stderr, "Error: Matrix dimension mismatch in copy\n");
        return;
    }
    
    if (mat_is_contiguous(src) && mat_is_contiguous(dst)) {
        memcpy(dst->data, src->data, (size_t)src->rows * src->cols * sizeof(f32));
    } else {
        for (u32 i = 0; i < src->rows; i++) {
            for (u32 j = 0; j < src->cols; j++) {
                mat_set_at(dst, i, j, mat_at(src, i, j));
            }
        }
    }
}

/**
 * 矩阵加法: result = a + b
 */
void mat_add(const matrix* a, const matrix* b, matrix* result) {
    if (!mat_is_valid(a) || !mat_is_valid(b) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in add\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols || 
        a->rows != result->rows || a->cols != result->cols) {
        fprintf(stderr, "Error: Dimension mismatch in add\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b) && mat_is_contiguous(result)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            result->data[i] = a->data[i] + b->data[i];
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_a = mat_at(a, i, j);
                f32 val_b = mat_at(b, i, j);
                mat_set_at(result, i, j, val_a + val_b);
            }
        }
    }
}

/**
 * 矩阵减法: result = a - b
 */
void mat_sub(const matrix* a, const matrix* b, matrix* result) {
    if (!mat_is_valid(a) || !mat_is_valid(b) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in sub\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols || 
        a->rows != result->rows || a->cols != result->cols) {
        fprintf(stderr, "Error: Dimension mismatch in sub\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b) && mat_is_contiguous(result)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            result->data[i] = a->data[i] - b->data[i];
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_a = mat_at(a, i, j);
                f32 val_b = mat_at(b, i, j);
                mat_set_at(result, i, j, val_a - val_b);
            }
        }
    }
}

/**
 * 矩阵标量乘法: result = a * scalar
 */
void mat_mul_scalar(const matrix* a, f32 scalar, matrix* result) {
    if (!mat_is_valid(a) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in mul_scalar\n");
        return;
    }
    
    if (a->rows != result->rows || a->cols != result->cols) {
        fprintf(stderr, "Error: Dimension mismatch in mul_scalar\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(result)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            result->data[i] = a->data[i] * scalar;
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_a = mat_at(a, i, j);
                mat_set_at(result, i, j, val_a * scalar);
            }
        }
    }
}

/**
 * 矩阵乘法: result = a * b
 * a: [m x k], b: [k x n], result: [m x n]
 */
void mat_matmul(const matrix* a, const matrix* b, matrix* result) {
    if (!mat_is_valid(a) || !mat_is_valid(b) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in matmul\n");
        return;
    }
    
    if (a->cols != b->rows || a->rows != result->rows || b->cols != result->cols) {
        fprintf(stderr, "Error: Dimension mismatch in matmul: a(%u,%u) * b(%u,%u) -> result(%u,%u)\n",
                a->rows, a->cols, b->rows, b->cols, result->rows, result->cols);
        return;
    }
    
    u32 m = a->rows;
    u32 k = a->cols;
    u32 n = b->cols;
    
    // 优化：连续内存路径
    if (mat_is_contiguous(a) && mat_is_contiguous(b) && mat_is_contiguous(result)) {
        /* 更缓存友好的实现：先转置 B 为 (n x k)，使内层循环顺序访问内存 */
        f32* bt = NULL;
        int bt_from_arena = 0;
        mem_arena* tmp_arena = NULL;
        if (result && result->arena) tmp_arena = result->arena;
        else if (a && a->arena) tmp_arena = a->arena;

        if (tmp_arena) {
            bt = ARENA_ALLOC_ARRAY(tmp_arena, f32, (size_t)k * n);
            bt_from_arena = 1;
        } else {
            bt = (f32*)malloc((size_t)k * n * sizeof(f32));
        }
        if (bt) {
            for (u32 l = 0; l < k; l++) {
                for (u32 j = 0; j < n; j++) {
                    bt[j * k + l] = b->data[l * n + j];
                }
            }

            for (u32 i = 0; i < m; i++) {
                for (u32 j = 0; j < n; j++) {
                    f32 sum = 0.0f;
                    f32* bt_row = bt + (size_t)j * k;
                    f32* a_row = a->data + (size_t)i * k;
                    for (u32 l = 0; l < k; l++) {
                        sum += a_row[l] * bt_row[l];
                    }
                    result->data[i * n + j] = sum;
                }
            }

            if (!bt_from_arena) free(bt);
        } else {
            /* 回退到简单实现（如果内存分配失败） */
            for (u32 i = 0; i < m; i++) {
                for (u32 j = 0; j < n; j++) {
                    f32 sum = 0.0f;
                    for (u32 l = 0; l < k; l++) {
                        sum += a->data[i * k + l] * b->data[l * n + j];
                    }
                    result->data[i * n + j] = sum;
                }
            }
        }
    } else {
        // 通用路径
        for (u32 i = 0; i < m; i++) {
            for (u32 j = 0; j < n; j++) {
                f32 sum = 0.0f;
                for (u32 l = 0; l < k; l++) {
                    sum += mat_at(a, i, l) * mat_at(b, l, j);
                }
                mat_set_at(result, i, j, sum);
            }
        }
    }
}

// === 统计函数实现 ===

/**
 * 计算矩阵所有元素之和
 */
f32 mat_sum(const matrix* m) {
    if (!mat_is_valid(m)) return 0.0f;
    
    f32 sum = 0.0f;
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            sum += m->data[i];
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                sum += mat_at(m, i, j);
            }
        }
    }
    return sum;
}

/**
 * 计算矩阵所有元素平方和
 */
f32 mat_sum_of_squares(const matrix* m) {
    if (!mat_is_valid(m)) return 0.0f;
    
    f32 sum_sq = 0.0f;
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            f32 val = m->data[i];
            sum_sq += val * val;
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = mat_at(m, i, j);
                sum_sq += val * val;
            }
        }
    }
    return sum_sq;
}

/**
 * 计算矩阵均值
 */
f32 mat_mean(const matrix* m) {
    if (!mat_is_valid(m)) return 0.0f;
    size_t size = mat_size(m);
    return size > 0 ? mat_sum(m) / (f32)size : 0.0f;
}

/**
 * 计算矩阵方差
 */
f32 mat_variance(const matrix* m) {
    if (!mat_is_valid(m)) return 0.0f;
    size_t size = mat_size(m);
    if (size <= 1) return 0.0f;
    
    f32 mean = mat_mean(m);
    f32 sum_sq_diff = 0.0f;
    
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            f32 diff = m->data[i] - mean;
            sum_sq_diff += diff * diff;
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 diff = mat_at(m, i, j) - mean;
                sum_sq_diff += diff * diff;
            }
        }
    }
    
    return sum_sq_diff / (f32)(size - 1);
}

// === 聚合操作实现 ===

/**
 * 按行求和: [rows x cols] -> [rows x 1]
 */
void mat_sum_rows(const matrix* m, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in sum_rows\n");
        return;
    }
    
    if (result->cols != 1 || result->rows != m->rows) {
        fprintf(stderr, "Error: Result dimension mismatch in sum_rows\n");
        return;
    }
    
    if (mat_is_contiguous(m) && mat_is_contiguous(result)) {
        for (u32 i = 0; i < m->rows; i++) {
            f32 sum = 0.0f;
            for (u32 j = 0; j < m->cols; j++) {
                sum += m->data[i * m->cols + j];
            }
            result->data[i] = sum;
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            f32 sum = 0.0f;
            for (u32 j = 0; j < m->cols; j++) {
                sum += mat_at(m, i, j);
            }
            mat_set_at(result, i, 0, sum);
        }
    }
}

/**
 * 按列求和: [rows x cols] -> [1 x cols]
 */
void mat_sum_cols(const matrix* m, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in sum_cols\n");
        return;
    }
    
    if (result->rows != 1 || result->cols != m->cols) {
        fprintf(stderr, "Error: Result dimension mismatch in sum_cols\n");
        return;
    }
    
    if (mat_is_contiguous(m) && mat_is_contiguous(result)) {
        // 初始化为0
        for (u32 j = 0; j < m->cols; j++) {
            result->data[j] = 0.0f;
        }
        
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                result->data[j] += m->data[i * m->cols + j];
            }
        }
    } else {
        for (u32 j = 0; j < m->cols; j++) {
            f32 sum = 0.0f;
            for (u32 i = 0; i < m->rows; i++) {
                sum += mat_at(m, i, j);
            }
            mat_set_at(result, 0, j, sum);
        }
    }
}

// === 广播操作实现 ===

/**
 * 列向量广播加法: result[i,j] = m[i,j] + col_vec[i,0]
 */
void mat_broadcast_add_col(const matrix* m, const matrix* col_vec, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(col_vec) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in broadcast_add_col\n");
        return;
    }
    
    if (col_vec->cols != 1 || col_vec->rows != m->rows ||
        result->rows != m->rows || result->cols != m->cols) {
        fprintf(stderr, "Error: Dimension mismatch in broadcast_add_col\n");
        return;
    }
    
    if (mat_is_contiguous(m) && mat_is_contiguous(col_vec) && mat_is_contiguous(result)) {
        for (u32 i = 0; i < m->rows; i++) {
            f32 bias = col_vec->data[i];
            for (u32 j = 0; j < m->cols; j++) {
                result->data[i * m->cols + j] = m->data[i * m->cols + j] + bias;
            }
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            f32 bias = mat_at(col_vec, i, 0);
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = mat_at(m, i, j);
                mat_set_at(result, i, j, val + bias);
            }
        }
    }
}

/**
 * 列向量广播减法: result[i,j] = m[i,j] - col_vec[i,0]
 */
void mat_broadcast_sub_col(const matrix* m, const matrix* col_vec, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(col_vec) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in broadcast_sub_col\n");
        return;
    }
    
    if (col_vec->cols != 1 || col_vec->rows != m->rows ||
        result->rows != m->rows || result->cols != m->cols) {
        fprintf(stderr, "Error: Dimension mismatch in broadcast_sub_col\n");
        return;
    }
    
    if (mat_is_contiguous(m) && mat_is_contiguous(col_vec) && mat_is_contiguous(result)) {
        for (u32 i = 0; i < m->rows; i++) {
            f32 bias = col_vec->data[i];
            for (u32 j = 0; j < m->cols; j++) {
                result->data[i * m->cols + j] = m->data[i * m->cols + j] - bias;
            }
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            f32 bias = mat_at(col_vec, i, 0);
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = mat_at(m, i, j);
                mat_set_at(result, i, j, val - bias);
            }
        }
    }
}

void mat_broadcast_add_row(const matrix* m, const matrix* row_vec, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(row_vec) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in broadcast_add_row\n");
        return;
    }

    if (row_vec->rows != 1 || row_vec->cols != m->cols ||
        result->rows != m->rows || result->cols != m->cols) {
        fprintf(stderr, "Error: Dimension mismatch in broadcast_add_row\n");
        return;
    }

    if (mat_is_contiguous(m) && mat_is_contiguous(row_vec) && mat_is_contiguous(result)) {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                result->data[i * m->cols + j] = m->data[i * m->cols + j] + row_vec->data[j];
            }
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 bias = mat_at(row_vec, 0, j);
                f32 val = mat_at(m, i, j);
                mat_set_at(result, i, j, val + bias);
            }
        }
    }
}

void mat_broadcast_sub_row(const matrix* m, const matrix* row_vec, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(row_vec) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in broadcast_sub_row\n");
        return;
    }

    if (row_vec->rows != 1 || row_vec->cols != m->cols ||
        result->rows != m->rows || result->cols != m->cols) {
        fprintf(stderr, "Error: Dimension mismatch in broadcast_sub_row\n");
        return;
    }

    if (mat_is_contiguous(m) && mat_is_contiguous(row_vec) && mat_is_contiguous(result)) {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                result->data[i * m->cols + j] = m->data[i * m->cols + j] - row_vec->data[j];
            }
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 bias = mat_at(row_vec, 0, j);
                f32 val = mat_at(m, i, j);
                mat_set_at(result, i, j, val - bias);
            }
        }
    }
}

// === 元素级操作实现 ===

/**
 * 元素级乘法: result = a .* b
 */
void mat_elementwise_mul(const matrix* a, const matrix* b, matrix* result) {
    if (!mat_is_valid(a) || !mat_is_valid(b) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in elementwise_mul\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols || 
        a->rows != result->rows || a->cols != result->cols) {
        fprintf(stderr, "Error: Dimension mismatch in elementwise_mul\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b) && mat_is_contiguous(result)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            result->data[i] = a->data[i] * b->data[i];
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_a = mat_at(a, i, j);
                f32 val_b = mat_at(b, i, j);
                mat_set_at(result, i, j, val_a * val_b);
            }
        }
    }
}

/**
 * 原地元素级乘法: a = a .* b
 */
void mat_elementwise_mul_inplace(matrix* a, const matrix* b) {
    if (!mat_is_valid(a) || !mat_is_valid(b)) {
        fprintf(stderr, "Error: Invalid matrices in elementwise_mul_inplace\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in elementwise_mul_inplace\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            a->data[i] *= b->data[i];
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_a = mat_at(a, i, j);
                f32 val_b = mat_at(b, i, j);
                mat_set_at(a, i, j, val_a * val_b);
            }
        }
    }
}

void mat_elementwise_div(const matrix* a, const matrix* b, matrix* result) {
    if (!mat_is_valid(a) || !mat_is_valid(b) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in elementwise_div\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols || 
        a->rows != result->rows || a->cols != result->cols) {
        fprintf(stderr, "Error: Dimension mismatch in elementwise_div\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b) && mat_is_contiguous(result)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            f32 denom = b->data[i];
            if (denom == 0.0f) {
                fprintf(stderr, "Warning: Division by zero in elementwise_div at index %zu\n", i);
                result->data[i] = 0.0f;
            } else {
                result->data[i] = a->data[i] / denom;
            }
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_a = mat_at(a, i, j);
                f32 val_b = mat_at(b, i, j);
                if (val_b == 0.0f) {
                    fprintf(stderr, "Warning: Division by zero in elementwise_div at (%u,%u)\n", i, j);
                    mat_set_at(result, i, j, 0.0f);
                } else {
                    mat_set_at(result, i, j, val_a / val_b);
                }
            }
        }
    }
}

void mat_elementwise_div_inplace(matrix* a, const matrix* b) {
    if (!mat_is_valid(a) || !mat_is_valid(b)) {
        fprintf(stderr, "Error: Invalid matrices in elementwise_div_inplace\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in elementwise_div_inplace\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            f32 denom = b->data[i];
            if (denom == 0.0f) {
                fprintf(stderr, "Warning: Division by zero in elementwise_div_inplace at index %zu\n", i);
                a->data[i] = 0.0f;
            } else {
                a->data[i] /= denom;
            }
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_b = mat_at(b, i, j);
                if (val_b == 0.0f) {
                    fprintf(stderr, "Warning: Division by zero in elementwise_div_inplace at (%u,%u)\n", i, j);
                    mat_set_at(a, i, j, 0.0f);
                } else {
                    f32 val_a = mat_at(a, i, j);
                    mat_set_at(a, i, j, val_a / val_b);
                }
            }
        }
    }
}

// === 转置操作实现 ===

/**
 * 矩阵转置: result = m^T
 */
void mat_transpose(const matrix* m, matrix* result) {
    if (!mat_is_valid(m) || !mat_is_valid(result)) {
        fprintf(stderr, "Error: Invalid matrices in transpose\n");
        return;
    }
    
    if (result->rows != m->cols || result->cols != m->rows) {
        fprintf(stderr, "Error: Result dimension mismatch in transpose\n");
        return;
    }
    
    if (mat_is_contiguous(m) && mat_is_contiguous(result)) {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                result->data[j * m->rows + i] = m->data[i * m->cols + j];
            }
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = mat_at(m, i, j);
                mat_set_at(result, j, i, val);
            }
        }
    }
}

// === 随机数生成实现 ===

/**
 * 简单线性同余生成器（用于可重现的随机数）
 */
static u32 lcg_rand(u64* seed) {
    *seed = (*seed * 1103515245ULL + 12345ULL) & 0x7fffffffULL;
    return (u32)(*seed);
}

/**
 * 生成[0,1)范围的随机浮点数
 */
static f32 rand_float(u64* seed) {
    return (f32)lcg_rand(seed) / (f32)0x7fffffff;
}

/**
 * Box-Muller变换生成正态分布随机数
 */
static f32 rand_normal(u64* seed, f32 mean, f32 stddev) {
    static bool has_spare = false;
    static f32 spare;
    
    if (has_spare) {
        has_spare = false;
        return mean + stddev * spare;
    }
    
    f32 u, v, s;
    do {
        u = 2.0f * rand_float(seed) - 1.0f;
        v = 2.0f * rand_float(seed) - 1.0f;
        s = u * u + v * v;
    } while (s >= 1.0f || s == 0.0f);
    
    f32 mul = sqrtf(-2.0f * logf(s) / s);
    spare = v * mul;
    has_spare = true;
    
    return mean + stddev * u * mul;
}

/**
 * 均匀分布随机初始化
 */
void mat_random_uniform(matrix* m, f32 min_val, f32 max_val, u64* seed) {
    if (!mat_is_valid(m)) return;
    
    f32 range = max_val - min_val;
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            m->data[i] = min_val + range * rand_float(seed);
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = min_val + range * rand_float(seed);
                mat_set_at(m, i, j, val);
            }
        }
    }
}

/**
 * 正态分布随机初始化
 */
void mat_random_normal(matrix* m, f32 mean, f32 stddev, u64* seed) {
    if (!mat_is_valid(m)) return;
    
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            m->data[i] = rand_normal(seed, mean, stddev);
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = rand_normal(seed, mean, stddev);
                mat_set_at(m, i, j, val);
            }
        }
    }
}

// === 激活函数实现 ===

/**
 * ReLU激活函数: output = max(0, input)
 */
void mat_relu(const matrix* input, matrix* output) {
    if (!mat_is_valid(input) || !mat_is_valid(output)) {
        fprintf(stderr, "Error: Invalid matrices in relu\n");
        return;
    }
    
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Error: Dimension mismatch in relu\n");
        return;
    }
    
    if (mat_is_contiguous(input) && mat_is_contiguous(output)) {
        size_t elements = mat_size(input);
        for (size_t i = 0; i < elements; i++) {
            output->data[i] = fmaxf(0.0f, input->data[i]);
        }
    } else {
        for (u32 i = 0; i < input->rows; i++) {
            for (u32 j = 0; j < input->cols; j++) {
                f32 val = mat_at(input, i, j);
                mat_set_at(output, i, j, fmaxf(0.0f, val));
            }
        }
    }
}

/**
 * Sigmoid激活函数: output = 1 / (1 + exp(-input))
 */
void mat_sigmoid(const matrix* input, matrix* output) {
    if (!mat_is_valid(input) || !mat_is_valid(output)) {
        fprintf(stderr, "Error: Invalid matrices in sigmoid\n");
        return;
    }
    
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Error: Dimension mismatch in sigmoid\n");
        return;
    }
    
    // 数值稳定性：避免exp溢出
    if (mat_is_contiguous(input) && mat_is_contiguous(output)) {
        size_t elements = mat_size(input);
        for (size_t i = 0; i < elements; i++) {
            f32 x = input->data[i];
            if (x >= 0) {
                f32 exp_neg_x = fast_exp(-x);
                output->data[i] = 1.0f / (1.0f + exp_neg_x);
            } else {
                f32 exp_x = fast_exp(x);
                output->data[i] = exp_x / (1.0f + exp_x);
            }
        }
    } else {
        for (u32 i = 0; i < input->rows; i++) {
            for (u32 j = 0; j < input->cols; j++) {
                f32 x = mat_at(input, i, j);
                f32 result;
                if (x >= 0) {
                    f32 exp_neg_x = fast_exp(-x);
                    result = 1.0f / (1.0f + exp_neg_x);
                } else {
                    f32 exp_x = fast_exp(x);
                    result = exp_x / (1.0f + exp_x);
                }
                mat_set_at(output, i, j, result);
            }
        }
    }
}

/**
 * Tanh激活函数: output = tanh(input)
 */
void mat_tanh(const matrix* input, matrix* output) {
    if (!mat_is_valid(input) || !mat_is_valid(output)) {
        fprintf(stderr, "Error: Invalid matrices in tanh\n");
        return;
    }
    
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Error: Dimension mismatch in tanh\n");
        return;
    }
    
    if (mat_is_contiguous(input) && mat_is_contiguous(output)) {
        size_t elements = mat_size(input);
        for (size_t i = 0; i < elements; i++) {
            output->data[i] = fast_tanh(input->data[i]);
        }
    } else {
        for (u32 i = 0; i < input->rows; i++) {
            for (u32 j = 0; j < input->cols; j++) {
                f32 val = mat_at(input, i, j);
                mat_set_at(output, i, j, fast_tanh(val));
            }
        }
    }
}

/**
 * Softmax激活函数（数值稳定版本）
 * output[i] = exp(input[i] - max_input) / sum(exp(input - max_input))
 */
void mat_softmax(const matrix* input, matrix* output) {
    if (!mat_is_valid(input) || !mat_is_valid(output)) {
        fprintf(stderr, "Error: Invalid matrices in softmax\n");
        return;
    }
    
    if (input->rows != output->rows || input->cols != output->cols) {
        fprintf(stderr, "Error: Dimension mismatch in softmax\n");
        return;
    }
    
    u32 batch_size = input->rows;
    u32 num_classes = input->cols;
    
    // 逐行处理（每个样本独立softmax）
    for (u32 i = 0; i < batch_size; i++) {
        // 找到最大值（数值稳定性）
        f32 max_val = -INFINITY;
        for (u32 j = 0; j < num_classes; j++) {
            f32 val = mat_at(input, i, j);
            if (val > max_val) max_val = val;
        }
        
        // 计算exp(input - max_val)
        f32 sum_exp = 0.0f;
        for (u32 j = 0; j < num_classes; j++) {
            f32 shifted_val = mat_at(input, i, j) - max_val;
            f32 exp_val = fast_exp(shifted_val);
            mat_set_at(output, i, j, exp_val);
            sum_exp += exp_val;
        }
        
        // 归一化
        if (sum_exp > 0.0f) {
            for (u32 j = 0; j < num_classes; j++) {
                f32 val = mat_at(output, i, j);
                mat_set_at(output, i, j, val / sum_exp);
            }
        }
    }
}


void mat_scale_inplace(matrix* m, f32 scalar) {
    if (!mat_is_valid(m)) return;
    
    if (mat_is_contiguous(m)) {
        size_t elements = mat_size(m);
        for (size_t i = 0; i < elements; i++) {
            m->data[i] *= scalar;
        }
    } else {
        for (u32 i = 0; i < m->rows; i++) {
            for (u32 j = 0; j < m->cols; j++) {
                f32 val = mat_at(m, i, j);
                mat_set_at(m, i, j, val * scalar);
            }
        }
    }
}

void mat_add_inplace(matrix* a, const matrix* b) {
    if (!mat_is_valid(a) || !mat_is_valid(b)) {
        fprintf(stderr, "Error: Invalid matrices in add_inplace\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in add_inplace\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            a->data[i] += b->data[i];
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_b = mat_at(b, i, j);
                f32 val_a = mat_at(a, i, j);
                mat_set_at(a, i, j, val_a + val_b);
            }
        }
    }
}

void mat_sub_inplace(matrix* a, const matrix* b) {
    if (!mat_is_valid(a) || !mat_is_valid(b)) {
        fprintf(stderr, "Error: Invalid matrices in sub_inplace\n");
        return;
    }
    
    if (a->rows != b->rows || a->cols != b->cols) {
        fprintf(stderr, "Error: Dimension mismatch in sub_inplace\n");
        return;
    }
    
    if (mat_is_contiguous(a) && mat_is_contiguous(b)) {
        size_t elements = mat_size(a);
        for (size_t i = 0; i < elements; i++) {
            a->data[i] -= b->data[i];
        }
    } else {
        for (u32 i = 0; i < a->rows; i++) {
            for (u32 j = 0; j < a->cols; j++) {
                f32 val_b = mat_at(b, i, j);
                f32 val_a = mat_at(a, i, j);
                mat_set_at(a, i, j, val_a - val_b);
            }
        }
    }
}

void mat_axpy_inplace(matrix* y, f32 alpha, const matrix* x) {
    if (!mat_is_valid(y) || !mat_is_valid(x)) {
        fprintf(stderr, "Error: Invalid matrices in axpy_inplace\n");
        return;
    }
    
    if (y->rows != x->rows || y->cols != x->cols) {
        fprintf(stderr, "Error: Dimension mismatch in axpy_inplace\n");
        return;
    }
    
    if (mat_is_contiguous(y) && mat_is_contiguous(x)) {
        size_t elements = mat_size(y);
        for (size_t i = 0; i < elements; i++) {
            y->data[i] += alpha * x->data[i];
        }
    } else {
        for (u32 i = 0; i < y->rows; i++) {
            for (u32 j = 0; j < y->cols; j++) {
                f32 val_x = mat_at(x, i, j);
                f32 val_y = mat_at(y, i, j);
                mat_set_at(y, i, j, val_y + alpha * val_x);
            }
        }
    }
}