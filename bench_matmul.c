#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include "arena.h"
#include "matrix.h"

static double now_seconds() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main() {
    mem_arena arena;
    arena_init(&arena, 32 * 1024 * 1024); // 32MB

    mat_options opts = MAT_DEFAULT_OPTS;
    u64 seed = 123456789ULL;

    int sizes[] = {128, 256, 512};
    int nsizes = sizeof(sizes)/sizeof(sizes[0]);

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        printf("\nBenchmark: N=%d (square matrices)\n", N);

        matrix* A = mat_create(&arena, N, N, opts);
        matrix* B = mat_create(&arena, N, N, opts);
        matrix* C = mat_create(&arena, N, N, opts);
        if (!A || !B || !C) { fprintf(stderr, "alloc failed\n"); return 1; }

        // randomize
        mat_random_uniform(A, -1.0f, 1.0f, &seed);
        mat_random_uniform(B, -1.0f, 1.0f, &seed);

        int repeats = 3;
        double total = 0.0;
        for (int r = 0; r < repeats; r++) {
            double t0 = now_seconds();
            mat_matmul(A, B, C);
            double t1 = now_seconds();
            double dt = t1 - t0;
            printf(" run %d: %.6f s\n", r, dt);
            total += dt;
            arena_reset(&arena); // reset to reuse buffers
            // reallocate C to avoid reusing same pointer for next run
            A = mat_create(&arena, N, N, opts);
            B = mat_create(&arena, N, N, opts);
            C = mat_create(&arena, N, N, opts);
            mat_random_uniform(A, -1.0f, 1.0f, &seed);
            mat_random_uniform(B, -1.0f, 1.0f, &seed);
        }
        printf(" avg: %.6f s\n", total / repeats);

        arena_reset(&arena);
    }

    arena_free(&arena);
    return 0;
}
