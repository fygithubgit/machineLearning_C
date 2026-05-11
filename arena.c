#include "arena.h"
#include <stdio.h>
#include <stdlib.h>

void* mem_arena_alloc(mem_arena* arena, size_t size, size_t alignment) {
    if (!arena || alignment == 0) {
        fprintf(stderr, "Error: Invalid arena alloc parameters\n");
        return NULL;
    }

    if (!arena->memory) {
        fprintf(stderr, "Error: Arena memory is NULL\n");
        return NULL;
    }

    if (size == 0) {
        fprintf(stderr, "Error: Cannot allocate zero bytes\n");
        return NULL;
    }

    /* alignment must be power of two */
    if ((alignment & (alignment - 1)) != 0) {
        fprintf(stderr, "Error: alignment must be a power of two\n");
        return NULL;
    }

    if (arena->offset > arena->capacity) {
        fprintf(stderr, "Error: Arena offset exceeds capacity\n");
        return NULL;
    }

    /* 计算对齐后的地址（基于 arena->memory 的基地址） */
    uintptr_t base = (uintptr_t)arena->memory;
    uintptr_t addr = base + arena->offset;
    uintptr_t aligned_addr = (addr + alignment - 1) & ~(uintptr_t)(alignment - 1);
    size_t padding = aligned_addr - addr;

    /* 检查 padding 与 size 是否会超出剩余容量，避免无符号下的溢出 */
    if (padding > arena->capacity - arena->offset) {
        fprintf(stderr, "Error: Arena memory exhausted (padding)\n");
        return NULL;
    }

    size_t remaining = arena->capacity - arena->offset - padding;
    if (size > remaining) {
        fprintf(stderr, "Error: Arena memory exhausted\n");
        return NULL;
    }

    arena->offset += padding;
    void* ptr = arena->memory + arena->offset;
    arena->offset += size;

    return ptr;
}

//-----------
void arena_init(mem_arena* arena, size_t size) {
    // 验证参数
    if (!arena || size == 0) {
        fprintf(stderr, "Invalid arena or size parameter\n");
        return;
    }
    
    // 分配底层内存
    arena->memory = malloc(size);
    if (!arena->memory) {
        fprintf(stderr, "Failed to allocate arena memory of size: %zu\n", size);
        exit(EXIT_FAILURE);
    }
    
    // 设置初始状态
    arena->capacity = size;
    arena->offset = 0;
    
    printf("Arena initialized with %zu bytes\n", size);
}

void arena_free(mem_arena* arena) {
    if (arena && arena->memory) {
        free(arena->memory);
        arena->memory = NULL;
        arena->capacity = 0;
        arena->offset = 0;
        printf("Arena memory freed\n");
    }
}

size_t arena_remaining(const mem_arena* arena) {
    if (!arena) return 0;
    return arena->capacity - arena->offset;
}

double arena_usage_percent(const mem_arena* arena) {
    if (!arena || arena->capacity == 0) return 0.0;
    return (double)arena->offset / (double)arena->capacity * 100.0;
}

mem_arena* arena_create_child(mem_arena* parent, size_t size) {
    if (!parent || size == 0) return NULL;
    
    // 在父arena中分配子arena结构体
    mem_arena* child = ARENA_ALLOC_STRUCT(parent, mem_arena);
    if (!child) return NULL;
    
    // 分配子arena的数据内存
    child->memory = mem_arena_alloc(parent, size, ARENA_DEFAULT_ALIGNMENT);
    if (!child->memory) return NULL;
    
    child->capacity = size;
    child->offset = 0;
    
    return child;
}

void mem_arena_reset(mem_arena* arena) {
    if (arena) {
        arena->offset = 0;
    }
}