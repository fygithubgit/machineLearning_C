#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>
#include <stdint.h>

typedef unsigned char       u8;
typedef unsigned short      u16;
typedef unsigned int        u32;
typedef unsigned long long  u64;
typedef signed char         i8;
typedef signed short        i16;
typedef signed int          i32;
typedef signed long long    i64;
typedef float               f32;
typedef double              f64;
typedef u8                  b8;  // 布尔类型
typedef u32                 b32;

#define MIN(a,b)            (((a)<(b))?(a):(b))
#define MAX(a,b)            (((a)>(b))?(a):(b))
#define CLAMP(x,min,max)    (MIN(MAX(x,min),max))
#define ALIGN_UP(value, alignment) \
    (((value) + (alignment) - 1) & ~((alignment) - 1))

#define ARENA_DEFAULT_ALIGNMENT 8


// Arena内存分配器结构
typedef struct mem_arena {
    unsigned char* memory;
    size_t capacity;
    size_t offset;
} mem_arena;

// 函数声明
void* mem_arena_alloc(mem_arena* arena, size_t size, size_t alignment);
void arena_init(mem_arena* arena, size_t size);
void arena_free(mem_arena* arena);
void mem_arena_reset(mem_arena* arena);
size_t arena_remaining(const mem_arena* arena);
double arena_usage_percent(const mem_arena* arena);

#define arena_reset mem_arena_reset


// 宏定义 - 便捷的分配接口
#define ARENA_ALLOC(arena, type, count, alignment) \
    ((type*)mem_arena_alloc((arena), sizeof(type) * (count), (alignment)))

#define ARENA_ALLOC_DEFAULT(arena, type, count) \
    ARENA_ALLOC((arena), type, (count), ARENA_DEFAULT_ALIGNMENT)

#define ARENA_ALLOC_STRUCT(arena, type) \
    ARENA_ALLOC_DEFAULT((arena), type, 1)

#define ARENA_ALLOC_ARRAY(arena, type, count) \
    ARENA_ALLOC_DEFAULT((arena), type, (count))
   
mem_arena* arena_create_child(mem_arena* parent, size_t size);
   

#endif