#include <stdio.h>
#include <stdlib.h>
#include "arena.h"
#include "matrix.h"
#include "layer.h"
#include <math.h>

int main() {
    mem_arena arena;
    arena_init(&arena, 8 * 1024 * 1024); // 8MB arena for test

    // 创建一个 dense 层：input_size=3, output_size=2
    layer* l = layer_create_dense(&arena, 3, 2, true);
    if (!l) { fprintf(stderr, "Failed to create layer\n"); return 1; }

    // 初始化权重为1, bias为0
    if (l->weights) mat_fill(l->weights, 1.0f);
    if (l->bias) mat_fill(l->bias, 0.0f);

    // 创建输入 [batch=2 x 3]
    mat_options opts = MAT_DEFAULT_OPTS;
    matrix* input = mat_create(&arena, 2, 3, opts);
    if (!input) { fprintf(stderr, "Failed to create input\n"); return 1; }
    // set input rows: [1,2,3] and [4,5,6]
    for (u32 i = 0; i < 2; i++) {
        for (u32 j = 0; j < 3; j++) {
            mat_set_at(input, i, j, (f32)(i*3 + j + 1));
        }
    }

    forward_ctx* ctx = forward_ctx_create(&arena);
    matrix* out = layer_forward(l, input, ctx);
    if (!out) { fprintf(stderr, "Forward failed\n"); return 1; }

    printf("Forward output:\n");
    for (u32 i = 0; i < out->rows; i++) {
        for (u32 j = 0; j < out->cols; j++) {
            printf("%f ", mat_at(out, i, j));
        }
        printf("\n");
    }

    // 验证前向输出
    float expected_out[2][2] = {{6.0f, 6.0f}, {15.0f, 15.0f}};
    for (u32 i = 0; i < out->rows; i++) {
        for (u32 j = 0; j < out->cols; j++) {
            float v = mat_at(out, i, j);
            if (fabsl(v - expected_out[i][j]) > 1e-6f) {
                fprintf(stderr, "Forward value mismatch at (%u,%u): %f != %f\n", i, j, v, expected_out[i][j]);
                return 2;
            }
        }
    }

    // 创建 grad_output 全1
    matrix* grad_out = mat_create(&arena, 2, 2, opts);
    mat_fill(grad_out, 1.0f);

    matrix* grad_input = layer_backward(l, grad_out, ctx);
    if (!grad_input) { fprintf(stderr, "Backward failed\n"); return 1; }

    printf("Grad weights (first row):\n");
    for (u32 k = 0; k < l->grad_weights->cols; k++) {
        printf("%f ", l->grad_weights->data[0 * l->grad_weights->cols + k]);
    }
    printf("\n");

    printf("Grad bias:\n");
    for (u32 i = 0; i < l->grad_bias->rows; i++) {
        printf("%f \n", mat_at(l->grad_bias, i, 0));
    }

    // 验证反向梯度
    float expected_gw_row0[] = {5.0f, 7.0f, 9.0f};
    for (u32 k = 0; k < l->grad_weights->cols; k++) {
        float v = l->grad_weights->data[0 * l->grad_weights->cols + k];
        if (fabsl(v - expected_gw_row0[k]) > 1e-6f) {
            fprintf(stderr, "Grad weight mismatch at index %u: %f != %f\n", k, v, expected_gw_row0[k]);
            return 3;
        }
    }
    for (u32 i = 0; i < l->grad_bias->rows; i++) {
        float b = mat_at(l->grad_bias, i, 0);
        if (fabsl(b - 2.0f) > 1e-6f) {
            fprintf(stderr, "Grad bias mismatch at %u: %f != 2.0\n", i, b);
            return 4;
        }
    }

    printf("All checks passed.\n");

    arena_free(&arena);
    // === Softmax + cross-entropy forward/backward test ===
    arena_init(&arena, 8 * 1024 * 1024);

    layer* s = layer_create_softmax(&arena);
    if (!s) { fprintf(stderr, "Failed to create softmax layer\n"); return 1; }

    // input logits [2 x 3]
    matrix* logits = mat_create(&arena, 2, 3, opts);
    // sample 0: [1,2,3]; sample1: [2,4,6]
    float lv[2][3] = {{1.0f,2.0f,3.0f},{2.0f,4.0f,6.0f}};
    for (u32 i=0;i<2;i++) for (u32 j=0;j<3;j++) mat_set_at(logits,i,j,lv[i][j]);

    forward_ctx* sc = forward_ctx_create(&arena);
    matrix* probs = layer_forward(s, logits, sc);
    if (!probs) { fprintf(stderr, "Softmax forward failed\n"); return 1; }

    // targets (one-hot): sample0 class2 (index 2), sample1 class2
    matrix* target = mat_create(&arena, 2, 3, opts);
    mat_fill(target, 0.0f);
    mat_set_at(target, 0, 2, 1.0f);
    mat_set_at(target, 1, 2, 1.0f);

    // grad_output for cross-entropy (output - target)
    matrix* grad_out2 = mat_create(&arena, 2, 3, opts);
    for (u32 i=0;i<2;i++) for (u32 j=0;j<3;j++) mat_set_at(grad_out2,i,j, mat_at(probs,i,j) - mat_at(target,i,j));

    matrix* grad_in2 = layer_backward(s, grad_out2, sc);
    if (!grad_in2) { fprintf(stderr, "Softmax backward failed\n"); return 1; }

    // Validate softmax backward by comparing to explicit jacobian-vector product
    for (u32 i=0;i<2;i++) {
        for (u32 a=0;a<3;a++) {
            float expected = 0.0f;
            float out_a = mat_at(probs, i, a);
            for (u32 b=0;b<3;b++) {
                float out_b = mat_at(probs, i, b);
                float grad_out_b = mat_at(grad_out2, i, b);
                float J = out_a * ((a==b ? 1.0f : 0.0f) - out_b);
                expected += J * grad_out_b;
            }
            float got = mat_at(grad_in2, i, a);
            if (fabsl(got - expected) > 1e-6f) {
                fprintf(stderr, "Softmax grad mismatch at (%u,%u): %f != %f\n", i, a, got, expected);
                return 5;
            }
        }
    }

    printf("Softmax backward jacobian test passed.\n");

    arena_free(&arena);
    return 0;
}
