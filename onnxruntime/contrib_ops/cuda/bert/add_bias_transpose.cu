// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cu_inc/common.cuh"
#include "contrib_ops/cuda/bert/add_bias_transpose.h"

namespace onnxruntime {
namespace cuda {

struct __align__(8) Half4 {
  half2 x;
  half2 y;
};

__device__ __forceinline__ Half4 operator+(const Half4& a, const Half4& b) {
  Half4 r;
  r.x = a.x + b.x;
  r.y = a.y + b.y;
  return r;
}

__device__ __forceinline__ float2 operator+(const float2& a, const float2& b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float4 operator+(const float4& a, const float4& b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

struct Float8_ {
  float2 x;
  float2 y;
  float2 z;
  float2 w;
};

struct Float4_ {
  float2 x;
  float2 y;
};

template<typename T>
struct num_elems;
template<>
struct num_elems<float> {
    static constexpr int value = 1;
};
template<>
struct num_elems<float2> {
    static constexpr int value = 2;
};
template<>
struct num_elems<float4> {
    static constexpr int value = 4;
};
template<>
struct num_elems<Float4_> {
    static constexpr int value = 4;
};
template<>
struct num_elems<Float8_> {
    static constexpr int value = 8;
};

template<>
struct num_elems<uint32_t> {
    static constexpr int value = 2;
};
template<>
struct num_elems<uint2> {
    static constexpr int value = 4;
};
template<>
struct num_elems<uint4> {
    static constexpr int value = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
struct Vec_t {
    static constexpr int size = 0;
};

template<>
struct Vec_t<float> {
    using Type = float2;
    static constexpr int size = 2;
};

template<>
struct Vec_t<float2> {
    using Type = float4;
    static constexpr int size = 4;
};

template<>
struct Vec_t<float4> {
    using Type = Float8_;
    static constexpr int size = 8;
};

template<>
struct Vec_t<half> {
    using Type = uint32_t;
    static constexpr int size = 2;
};

template<>
struct Vec_t<half2> {
    using Type = uint2;
    static constexpr int size = 4;
};

template<>
struct Vec_t<Half4> {
    using Type = uint4;
    static constexpr int size = 8;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, float b)
{
    return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(float2 a, float2 b)
{
    float2 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 add(float4 a, float4 b)
{
    float4 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(Float8_ a, Float8_ b)
{
    return;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t add(uint16_t a, uint16_t b)
{
    uint16_t c;
    asm volatile("add.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t add(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 add(uint2 a, uint2 b)
{
    uint2 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 add(uint4 a, uint4 b)
{
    uint4 c;
    c.x = add(a.x, b.x);
    c.y = add(a.y, b.y);
    c.z = add(a.z, b.z);
    c.w = add(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint16_t float_to_half(float f)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if 0 && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800  // Is it better?
  float zero = 0.f;
  asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(zero), "f"(f));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f));
#endif
    return tmp.u16[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t float2_to_half2(float2 f)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
#else
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
    asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
#endif
    return tmp.u32;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float half_to_float(uint16_t h)
{
    float f;
    asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
    return f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 half2_to_float2(uint32_t v)
{
    uint16_t lo, hi;
    asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
    return make_float2(half_to_float(lo), half_to_float(hi));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float add(float a, uint16_t b)
{
    return a + half_to_float(b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 add(uint32_t a, float2 fb)
{
    float2 fa = half2_to_float2(a);
    return add(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ add(uint2 a, Float4_ fb)
{
    Float4_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ add(uint4 a, Float8_ fb)
{
    Float8_ fc;
    fc.x = add(a.x, fb.x);
    fc.y = add(a.y, fb.y);
    fc.z = add(a.z, fb.z);
    fc.w = add(a.w, fb.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t h0_h0(uint16_t a)
{
    uint32_t b;
    asm volatile("mov.b32 %0, {%1, %1};" : "=r"(b) : "h"(a));
    return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(float a, float b, float c)
{
    return a * b + c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float2 a, float2 b, float2 c)
{
    float2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(float a, float2 b, float2 c)
{
    float2 d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float4 a, float4 b, float4 c)
{
    float4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float4 fma(float a, float4 b, float4 c)
{
    float4 d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    d.z = fma(a, b.z, c.z);
    d.w = fma(a, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(float a, Float4_ b, Float4_ c)
{
    Float4_ d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(float a, Float8_ b, Float8_ c)
{
    Float8_ d;
    d.x = fma(a, b.x, c.x);
    d.y = fma(a, b.y, c.y);
    d.z = fma(a, b.z, c.z);
    d.w = fma(a, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint32_t fma(uint16_t a, uint32_t b, uint32_t c)
{
    return fma(h0_h0(a), b, c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint2 a, uint2 b, uint2 c)
{
    uint2 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint2 fma(uint16_t a, uint2 b, uint2 c)
{
    uint32_t s = h0_h0(a);
    uint2    d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint4 a, uint4 b, uint4 c)
{
    uint4 d;
    d.x = fma(a.x, b.x, c.x);
    d.y = fma(a.y, b.y, c.y);
    d.z = fma(a.z, b.z, c.z);
    d.w = fma(a.w, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ uint4 fma(uint16_t a, uint4 b, uint4 c)
{
    uint32_t s = h0_h0(a);
    uint4    d;
    d.x = fma(s, b.x, c.x);
    d.y = fma(s, b.y, c.y);
    d.z = fma(s, b.z, c.z);
    d.w = fma(s, b.w, c.w);
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float fma(uint16_t a, uint16_t b, float fc)
{
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return fa * fb + fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint32_t a, uint32_t b, float2 fc)
{
    float2 fa = half2_to_float2(a);
    float2 fb = half2_to_float2(b);
    return fma(fa, fb, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 fma(uint16_t a, uint32_t b, float2 fc)
{
    return fma(h0_h0(a), b, fc);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint2 a, uint2 b, Float4_ fc)
{
    Float4_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float4_ fma(uint16_t a, uint2 b, Float4_ fc)
{
    uint32_t s = h0_h0(a);
    Float4_  fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint4 a, uint4 b, Float8_ fc)
{
    Float8_ fd;
    fd.x = fma(a.x, b.x, fc.x);
    fd.y = fma(a.y, b.y, fc.y);
    fd.z = fma(a.z, b.z, fc.z);
    fd.w = fma(a.w, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ Float8_ fma(uint16_t a, uint4 b, Float8_ fc)
{
    uint32_t s = h0_h0(a);
    Float8_  fd;
    fd.x = fma(s, b.x, fc.x);
    fd.y = fma(s, b.y, fc.y);
    fd.z = fma(s, b.z, fc.z);
    fd.w = fma(s, b.w, fc.w);
    return fd;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Acc, typename A, typename B>
inline __device__ Acc mul(A a, B b)
{
    return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul<float, float>(float a, float b)
{
    return a * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float2 a, float2 b)
{
    float2 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(float a, float2 b)
{
    float2 c;
    c.x = a * b.x;
    c.y = a * b.y;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float4 a, float4 b)
{
    float4 c;
    c.x = a.x * b.x;
    c.y = a.y * b.y;
    c.z = a.z * b.z;
    c.w = a.w * b.w;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float4 mul(float a, float4 b)
{
    float4 c;
    c.x = a * b.x;
    c.y = a * b.y;
    c.z = a * b.z;
    c.w = a * b.w;
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(float a, Float8_ b)
{
    Float8_ c;
    c.x = make_float2(a * b.x.x, a * b.x.y);
    c.y = make_float2(a * b.y.x, a * b.y.y);
    c.z = make_float2(a * b.z.x, a * b.z.y);
    c.w = make_float2(a * b.w.x, a * b.w.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint16_t mul(uint16_t a, uint16_t b)
{
    uint16_t c;
    asm volatile("mul.f16 %0, %1, %2;\n" : "=h"(c) : "h"(a), "h"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint32_t mul(uint16_t a, uint32_t b)
{
    return mul<uint32_t, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint2 mul(uint2 a, uint2 b)
{
    uint2 c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint2 mul(uint16_t a, uint2 b)
{
    uint32_t s = h0_h0(a);
    uint2    c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(uint4 a, uint4 b)
{
    uint4 c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(a.x, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(a.y, b.y);
    c.z = mul<uint32_t, uint32_t, uint32_t>(a.z, b.z);
    c.w = mul<uint32_t, uint32_t, uint32_t>(a.w, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ uint4 mul(uint16_t a, uint4 b)
{
    uint32_t s = h0_h0(a);
    uint4    c;
    c.x = mul<uint32_t, uint32_t, uint32_t>(s, b.x);
    c.y = mul<uint32_t, uint32_t, uint32_t>(s, b.y);
    c.z = mul<uint32_t, uint32_t, uint32_t>(s, b.z);
    c.w = mul<uint32_t, uint32_t, uint32_t>(s, b.w);
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(uint16_t a, uint16_t b)
{
    float fa = half_to_float(a);
    float fb = half_to_float(b);
    return fa * fb;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float mul(uint16_t a, float b)
{
    return half_to_float(a) * b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint32_t a, uint32_t b)
{
    float2 fa = half2_to_float2(a);
    float2 fb = half2_to_float2(b);
    return mul<float2, float2, float2>(fa, fb);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ float2 mul(uint16_t a, uint32_t b)
{
    return mul<float2, uint32_t, uint32_t>(h0_h0(a), b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(uint2 a, uint2 b)
{
    Float4_ fc;
    fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float4_ mul(uint16_t a, uint2 b)
{
    uint32_t s = h0_h0(a);
    Float4_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint4 a, uint4 b)
{
    Float8_ fc;
    fc.x = mul<float2, uint32_t, uint32_t>(a.x, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(a.y, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(a.z, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(a.w, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
inline __device__ Float8_ mul(uint16_t a, uint4 b)
{
    uint32_t s = h0_h0(a);
    Float8_  fc;
    fc.x = mul<float2, uint32_t, uint32_t>(s, b.x);
    fc.y = mul<float2, uint32_t, uint32_t>(s, b.y);
    fc.z = mul<float2, uint32_t, uint32_t>(s, b.z);
    fc.w = mul<float2, uint32_t, uint32_t>(s, b.w);
    return fc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float v)
{
    return v;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float2 v)
{
    return v.x + v.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(float4 v)
{
    return v.x + v.y + v.z + v.w;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint16_t v)
{
    return half_to_float(v);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint32_t v)
{
    float2 tmp = half2_to_float2(v);
    return tmp.x + tmp.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint2 v)
{
    uint32_t c = add(v.x, v.y);
    return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(uint4 v)
{
#if 1
    uint32_t c = add(v.x, v.y);
    c          = add(c, v.z);
    c          = add(c, v.w);
#else
    uint32_t c = add(v.x, v.y);
    uint32_t d = add(v.z, v.w);
    c          = add(c, d);
#endif
    return sum(c);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float4_ v)
{
    return v.x.x + v.x.y + v.y.x + v.y.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float sum(Float8_ v)
{
    return v.x.x + v.x.y + v.y.x + v.y.y + v.z.x + v.z.y + v.w.x + v.w.y;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ float dot(T a, T b)
{
    return sum(mul<T, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename A, typename T>
inline __device__ float dot(T a, T b)
{
    return sum(mul<A, T, T>(a, b));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void zero(uint16_t& dst)
{
    dst = uint16_t(0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename T>
inline __device__ void zero(T& dst)
{
    constexpr int WORDS = sizeof(T) / 4;
    union {
        T        raw;
        uint32_t words[WORDS];
    } tmp;
#pragma unroll
    for (int ii = 0; ii < WORDS; ++ii) {
        tmp.words[ii] = 0u;
    }
    dst = tmp.raw;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ float2 rotary_embedding_coefficient(const int zid, const int rot_embed_dim, const float t_step)
{
    const float inv_freq = t_step / pow(10000.0f, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

inline __device__ float2 rotary_embedding_transform(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ uint32_t rotary_embedding_transform(const uint32_t v, const float2 coef)
{
    float2 fv     = half2_to_float2(v);
    float2 rot_fv = rotary_embedding_transform(fv, coef);
    return float2_to_half2(rot_fv);
}

inline __device__ void apply_rotary_embedding(float& q, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(float& q, float& k, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(Float8_& q, Float8_& k, int zid, int rot_embed_dim, int t_step)
{
    return;
}

inline __device__ void apply_rotary_embedding(float2& q, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q               = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(float2& q, float2& k, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q               = rotary_embedding_transform(q, coef);
    k               = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(float4& q, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_&   q_    = *reinterpret_cast<Float4_*>(&q);
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q_.x             = rotary_embedding_transform(q_.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q_.y             = rotary_embedding_transform(q_.y, coef1);
}

inline __device__ void apply_rotary_embedding(float4& q, float4& k, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }

    Float4_&   q_    = *reinterpret_cast<Float4_*>(&q);
    Float4_&   k_    = *reinterpret_cast<Float4_*>(&k);
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q_.x             = rotary_embedding_transform(q_.x, coef0);
    k_.x             = rotary_embedding_transform(k_.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q_.y             = rotary_embedding_transform(q_.y, coef1);
    k_.y             = rotary_embedding_transform(k_.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q               = rotary_embedding_transform(q, coef);
}

inline __device__ void apply_rotary_embedding(uint32_t& q, uint32_t& k, int tid, int rot_embed_dim, int t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = rotary_embedding_coefficient(2 * tid, rot_embed_dim, t_step);
    q               = rotary_embedding_transform(q, coef);
    k               = rotary_embedding_transform(k, coef);
}

inline __device__ void apply_rotary_embedding(uint2& q, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint2& q, uint2& k, int tid, int rot_embed_dim, int t_step)
{
    if (4 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(4 * tid, rot_embed_dim, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    k.x              = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(4 * tid + 2, rot_embed_dim, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
    k.y              = rotary_embedding_transform(k.y, coef1);
}

inline __device__ void apply_rotary_embedding(uint4& q, int tid, int rot_embed_dim, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z              = rotary_embedding_transform(q.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w              = rotary_embedding_transform(q.w, coef3);
}

inline __device__ void apply_rotary_embedding(uint4& q, uint4& k, int tid, int rot_embed_dim, int t_step)
{
    if (8 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef0 = rotary_embedding_coefficient(8 * tid, rot_embed_dim, t_step);
    q.x              = rotary_embedding_transform(q.x, coef0);
    k.x              = rotary_embedding_transform(k.x, coef0);
    const auto coef1 = rotary_embedding_coefficient(8 * tid + 2, rot_embed_dim, t_step);
    q.y              = rotary_embedding_transform(q.y, coef1);
    k.y              = rotary_embedding_transform(k.y, coef1);
    const auto coef2 = rotary_embedding_coefficient(8 * tid + 4, rot_embed_dim, t_step);
    q.z              = rotary_embedding_transform(q.z, coef2);
    k.z              = rotary_embedding_transform(k.z, coef2);
    const auto coef3 = rotary_embedding_coefficient(8 * tid + 6, rot_embed_dim, t_step);
    q.w              = rotary_embedding_transform(q.w, coef3);
    k.w              = rotary_embedding_transform(k.w, coef3);
}

template<typename Vec_T, typename T>
__device__ __inline__ void vec_from_smem_transpose(Vec_T& vec, T* smem, int transpose_idx, int smem_pitch);

template<>
__device__ __inline__ void vec_from_smem_transpose(float& vec, float* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(float4& vec, float2* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(Float8_& vec, float4* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint2& vec, half2* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint4& vec, Half4* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint32_t& vec, uint16_t* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
    tmp.u16[0] = smem[transpose_idx];
    tmp.u16[1] = smem[smem_pitch + transpose_idx];

    vec = tmp.u32;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint2& vec, uint16_t* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp_1, tmp_2;
    tmp_1.u32 = *reinterpret_cast<uint32_t*>(&smem[transpose_idx]);
    tmp_2.u32 = *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]);

    union {
        uint2    u32x2;
        uint16_t u16[4];
    } tmp_3;
    tmp_3.u16[0] = tmp_1.u16[0];
    tmp_3.u16[1] = tmp_2.u16[0];
    tmp_3.u16[2] = tmp_1.u16[1];
    tmp_3.u16[3] = tmp_2.u16[1];

    vec = tmp_3.u32x2;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint4& vec, uint16_t* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint64_t u64;
        uint16_t u16[4];
    } tmp_1, tmp_2;
    tmp_1.u64 = *reinterpret_cast<uint64_t*>(&smem[transpose_idx]);
    tmp_2.u64 = *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]);

    union {
        uint4    u32x4;
        uint16_t u16[8];
    } tmp_3;
    tmp_3.u16[0] = tmp_1.u16[0];
    tmp_3.u16[1] = tmp_2.u16[0];
    tmp_3.u16[2] = tmp_1.u16[1];
    tmp_3.u16[3] = tmp_2.u16[1];
    tmp_3.u16[4] = tmp_1.u16[2];
    tmp_3.u16[5] = tmp_2.u16[2];
    tmp_3.u16[6] = tmp_1.u16[3];
    tmp_3.u16[7] = tmp_2.u16[3];

    vec = tmp_3.u32x4;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(float4& vec, float* smem, int transpose_idx, int smem_pitch)
{
    vec.x = smem[transpose_idx];
    vec.z = smem[transpose_idx + 1];
    vec.y = smem[smem_pitch + transpose_idx];
    vec.w = smem[smem_pitch + transpose_idx + 1];
}

template<>
__device__ __inline__ void vec_from_smem_transpose(uint32_t& vec, half* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint32_t u32;
        half     u16[2];
    } tmp;
    tmp.u16[0] = smem[transpose_idx];
    tmp.u16[1] = smem[smem_pitch + transpose_idx];

    vec = tmp.u32;
}

template<>
__device__ __inline__ void vec_from_smem_transpose(float2& vec, float* smem, int transpose_idx, int smem_pitch)
{
    vec.x = smem[transpose_idx];
    vec.y = smem[smem_pitch + transpose_idx];
}

template<typename Vec_T, typename T>
__device__ __inline__ void write_smem_transpose(const Vec_T& vec, T* smem, int transpose_idx, int smem_pitch);

template<>
__device__ __inline__ void write_smem_transpose(const float& vec, float* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void write_smem_transpose(const float4& vec, float2* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void write_smem_transpose(const Float8_& vec, float4* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint2& vec, half2* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint4& vec, Half4* smem, int transpose_idx, int smem_pitch)
{
    return;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint4& vec, uint16_t* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint64_t u64;
        uint16_t u16[4];
    } tmp_1, tmp_2;

    union {
        uint4    u32x4;
        uint16_t u16[8];
    } tmp_3;
    tmp_3.u32x4  = vec;
    tmp_1.u16[0] = tmp_3.u16[0];
    tmp_2.u16[0] = tmp_3.u16[1];
    tmp_1.u16[1] = tmp_3.u16[2];
    tmp_2.u16[1] = tmp_3.u16[3];
    tmp_1.u16[2] = tmp_3.u16[4];
    tmp_2.u16[2] = tmp_3.u16[5];
    tmp_1.u16[3] = tmp_3.u16[6];
    tmp_2.u16[3] = tmp_3.u16[7];

    *reinterpret_cast<uint64_t*>(&smem[transpose_idx])              = tmp_1.u64;
    *reinterpret_cast<uint64_t*>(&smem[smem_pitch + transpose_idx]) = tmp_2.u64;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint2& vec, uint16_t* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp_1, tmp_2;

    union {
        uint2    u32x2;
        uint16_t u16[4];
    } tmp_3;
    tmp_3.u32x2  = vec;
    tmp_1.u16[0] = tmp_3.u16[0];
    tmp_2.u16[0] = tmp_3.u16[1];
    tmp_1.u16[1] = tmp_3.u16[2];
    tmp_2.u16[1] = tmp_3.u16[3];

    *reinterpret_cast<uint32_t*>(&smem[transpose_idx])              = tmp_1.u32;
    *reinterpret_cast<uint32_t*>(&smem[smem_pitch + transpose_idx]) = tmp_2.u32;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint32_t& vec, uint16_t* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint32_t u32;
        uint16_t u16[2];
    } tmp;
    tmp.u32 = vec;

    smem[transpose_idx]              = tmp.u16[0];
    smem[smem_pitch + transpose_idx] = tmp.u16[1];
}

template<>
__device__ __inline__ void write_smem_transpose(const float4& vec, float* smem, int transpose_idx, int smem_pitch)
{
    smem[transpose_idx]                  = vec.x;
    smem[transpose_idx + 1]              = vec.z;
    smem[smem_pitch + transpose_idx]     = vec.y;
    smem[smem_pitch + transpose_idx + 1] = vec.w;
}

template<>
__device__ __inline__ void write_smem_transpose(const uint32_t& vec, half* smem, int transpose_idx, int smem_pitch)
{
    union {
        uint32_t u32;
        half     u16[2];
    } tmp;

    tmp.u32                          = vec;
    smem[transpose_idx]              = tmp.u16[0];
    smem[smem_pitch + transpose_idx] = tmp.u16[1];
}

template<>
__device__ __inline__ void write_smem_transpose(const float2& vec, float* smem, int transpose_idx, int smem_pitch)
{
    smem[transpose_idx]              = vec.x;
    smem[smem_pitch + transpose_idx] = vec.y;
}

}  // namespace cuda
}  // namespace onnxruntime

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void AddBiasTransposeTrt(const T* input, const T* biases, T* output) {
  // Format 2 for TensorRT fused attention (N*H <= 1024)
  //     Input:  BxSxMxNxH
  //     Output: BxSxNxMxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  // This kernel could support hidden size up to 4 * 1024 when T is Half4 and input is half.

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int H = blockDim.x;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const int NH = N * H;
  const int offset = (b * S + s) * M * NH;
  const int in_offset = offset + m * NH + n * H;
  const int out_offset = offset + (n * M + m) * H;

  const int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtLarge(const int head_size, const T* input, const T* biases, T* output) {
  // Format 2 for TensorRT fused attention (N*H > 1024)
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;

  const int stride = blockDim.x;
  const int H = head_size;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const int NH = N * H;
  const int offset = (b * S + s) * M * NH;
  const int in_offset = offset + m * NH + n * H;
  const int out_offset = offset + (n * M + m) * H;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeTrt(const T* query, const T* key, const T* value, const T* biases, T* output) {
  // Separated Q/K/V inputs for TensorRT fused attention (N*H <= 1024)
  //     Q:  BxSxNxH
  //     K:  BxSxNxH
  //     V:  BxSxNxH
  //     Output: BxSxNxMxH
  // B is batch_size, S is sequence_length, M is number of matrices (3), N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int H = blockDim.x;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const T* input = (m == 0 ? query : (m == 1 ? key : value));
  const int NH = N * H;
  const int in_offset = (b * S + s) * NH + n * H;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  const int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtLarge(const int head_size,
                                         const T* query, const T* key, const T* value, const T* biases, T* output) {
  // Separated Q/K/V inputs for TensorRT fused attention (N*H > 1024)
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int H = head_size;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int M = gridDim.z;

  const T* input = (m == 0 ? query : (m == 1 ? key : value));
  const int NH = N * H;
  const int in_offset = (b * S + s) * NH + n * H;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtKV(const T* key, const T* value, const T* biases, T* output) {
  // Separated K/V inputs for TensorRT fused cross attention (N*H <= 1024)
  //     K:  BxSxNxH
  //     V:  BxSxNxH
  //     Output: BxSxNxMxH (packed KV, requires H = H_v)
  // B is batch_size, S is sequence_length, M is number of matrices (2), N is num_heads, H is head_size

  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int H = blockDim.x;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int NH = N * H;

  const int in_offset = (b * S + s) * NH + n * H;
  const T* input = (m == 0 ? key : value);

  constexpr int M = 2;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  const int h = threadIdx.x;
  if (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[(m + 1) * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeTrtKVLarge(const int head_size,
                                           const T* key, const T* value, const T* biases,
                                           T* output) {
  // Separated K/V inputs for TensorRT fused cross attention (N*H > 1024)
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int H = head_size;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int NH = N * H;

  const int in_offset = (b * S + s) * NH + n * H;
  const T* input = (m == 0 ? key : value);

  constexpr int M = 2;
  const int out_offset = (b * S + s) * M * NH + (n * M + m) * H;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[(m + 1) * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTransposeQKV(int M, const T* input, const T* biases, T* output, T* qkv_add_bias) {
  // Format 1 for unfused attention, or fused causal attention
  //     Input:  BxSxMxNxH
  //     Output: MxBxNxSxH
  //     qkv_add_bias: BxSxMxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * head_size + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * head_size + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    if (nullptr != qkv_add_bias) {
      qkv_add_bias[in_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    }
  }
}

template <typename T>
__global__ void AddBiasTransposeQKV(int M, const T* input, const T* biases, T* output, T* qkv_add_bias,
                                    const int rotary_embedding_dim, const int head_size, const int step,
                                    const int format) {
  // AddBiasTransposeQKV with rotary embedding
  // Format 1 for unfused attention, or fused causal attention
  //     Input:  BxSxMxNxH
  //     Output: MxBxNxSxH
  //     qkv_add_bias: BxSxMxNxH
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSxMxNxH
  //     Output: MxBxSxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = blockIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.z;

  const int seq_len = (gridDim.x == step) ? s : step;

  const int num_heads = gridDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.z;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  constexpr int vec_size = Vec_t<T>::size;
  using Vec_t = typename Vec_t<T>::Type;

  extern __shared__ __align__(sizeof(float2)) char smem_[];

  int tidx = threadIdx.x;

  if (tidx * vec_size < head_size) {
    const bool is_masked = tidx * vec_size >= head_size;

    const int src_q_idx = n * head_size + (s * M) * NH + b * NHS * M + tidx * vec_size;
    const int src_k_idx = n * head_size + (1 + s * M) * NH + b * NHS * M + tidx * vec_size;
    const int src_v_idx = n * head_size + (2 + s * M) * NH + b * NHS * M + tidx * vec_size;

    Vec_t q, k, v;
    Vec_t q_bias, k_bias, v_bias;

    if (!is_masked) {
      q = *reinterpret_cast<const Vec_t*>(&input[src_q_idx]);
      k = *reinterpret_cast<const Vec_t*>(&input[src_k_idx]);
      v = *reinterpret_cast<const Vec_t*>(&input[src_v_idx]);

      q_bias = *reinterpret_cast<const Vec_t*>(&biases[n * H + tidx * vec_size]);
      k_bias = *reinterpret_cast<const Vec_t*>(&biases[NH + n * H + tidx * vec_size]);
      v_bias = *reinterpret_cast<const Vec_t*>(&biases[2 * NH + n * H + tidx * vec_size]);
    }

    q = add(q, q_bias);
    k = add(k, k_bias);
    v = add(v, v_bias);

    const bool do_rotary = !is_masked && vec_size * tidx < rotary_embedding_dim;

    T* q_smem = reinterpret_cast<T*>(smem_);
    T* k_smem = q_smem + rotary_embedding_dim;

    const int half_rotary_dim = rotary_embedding_dim / 2;
    const int half_idx        = (tidx * vec_size) / half_rotary_dim;
    const int intra_half_idx  = (tidx * vec_size) % half_rotary_dim;
    const int smem_pitch      = half_rotary_dim;

    if (do_rotary) {
      *reinterpret_cast<Vec_t*>(q_smem + half_idx * smem_pitch + intra_half_idx) = q;
      *reinterpret_cast<Vec_t*>(k_smem + half_idx * smem_pitch + intra_half_idx) = k;
    }

    __syncthreads();

    const int transpose_idx = half_idx * (half_rotary_dim / 2) + intra_half_idx / 2;
    constexpr int tidx_factor = vec_size / 2;

    if (do_rotary) {
      vec_from_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
      vec_from_smem_transpose(k, k_smem, transpose_idx, smem_pitch);

      apply_rotary_embedding(q, k, transpose_idx / tidx_factor, rotary_embedding_dim, seq_len);

      write_smem_transpose(q, q_smem, transpose_idx, smem_pitch);
      write_smem_transpose(k, k_smem, transpose_idx, smem_pitch);
    }

    __syncthreads();

    if (do_rotary) {
      q = *reinterpret_cast<Vec_t*>(q_smem + half_idx * smem_pitch + intra_half_idx);
      k = *reinterpret_cast<Vec_t*>(k_smem + half_idx * smem_pitch + intra_half_idx);
    }

    int dest_q_idx;
    int dest_k_idx;
    int dest_v_idx;

    // Format 1
    if (format == 1) {
      dest_q_idx = s * head_size + n * sequence_length * H + b * NHS + tidx * vec_size;
      dest_k_idx = s * head_size + n * sequence_length * H + b * NHS + NHS * batch_size + tidx * vec_size;
      dest_v_idx = s * head_size + n * sequence_length * H + b * NHS + 2 * NHS * batch_size + tidx * vec_size;
    }

    if (format == 3) {
      dest_q_idx = n * H + s * NH + b * NHS + tidx * vec_size;
      dest_k_idx = n * H + s * NH + b * NHS + NHS * batch_size + tidx * vec_size;
      dest_v_idx = n * H + s * NH + b * NHS + 2 * NHS * batch_size + tidx * vec_size;
    }

    if (!is_masked) {
      *reinterpret_cast<Vec_t*>(&output[dest_q_idx]) = q;
      *reinterpret_cast<Vec_t*>(&output[dest_k_idx]) = k;
      *reinterpret_cast<Vec_t*>(&output[dest_v_idx]) = v;

      if (nullptr != qkv_add_bias) {
        *reinterpret_cast<Vec_t*>(&qkv_add_bias[src_q_idx]) = q;
        *reinterpret_cast<Vec_t*>(&qkv_add_bias[src_k_idx]) = k;
        *reinterpret_cast<Vec_t*>(&qkv_add_bias[src_v_idx]) = v;
      }
    }
  }
}

// this suppose 3 matrix in total
template <typename T>
__global__ void AddBiasTransposeQKV(const T* input, const T* biases, T* output, int v_head_size) {
  // Format 1 for unfused attention
  //     Input:  BxSx(NxH + NxH + NxH_v)  (Packed QKV where K and V has different hidden sizes)
  //     Output: BxNxSxH + BxNxSxH + BxNxSxH_v
  // B is batch_size, S is sequence_length, N is num_heads, H is qk_head_size, H_v is v_head_size
  int n = threadIdx.y;        // head_num_id
  int s = blockIdx.x;         // sequence_id
  int b = blockIdx.y;         // batch_id
  int m = blockIdx.z;         // matrix id (Q=0, K=1, V=2)
  const int h = threadIdx.x;  // head_element_id

  const int qk_head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int head_size = (m == 2 ? v_head_size : qk_head_size);

  const int total_head_size = num_heads * (qk_head_size + qk_head_size + v_head_size);

  int in_offset;
  int out_offset;
  int bias_offset;
  in_offset = b * (total_head_size * sequence_length) +  // B
              s * (total_head_size) +                    // S
              m * (qk_head_size * num_heads) +           // M
              n * head_size +                            // N
              h;                                         // H

  out_offset = m * (num_heads * qk_head_size * sequence_length * batch_size) +  // M
               b * (num_heads * head_size * sequence_length) +                  // B
               n * (sequence_length * head_size) +                              // N
               s * (head_size) +                                                // S
               h;                                                               // H

  bias_offset = m * (num_heads * qk_head_size) +  // M
                n * (head_size) +                 // N
                h;                                // H

  if (h < head_size) {
    output[out_offset] = input[in_offset] + biases[bias_offset];
  }
}

template <typename T>
__global__ void AddBiasTransposeQKVLarge(const int head_size, const T* input, const T* biases, T* output,
                                         T* qkv_add_bias, const int M) {
  // Format 1 for unfused attention (N*H > 1024), or fused causal attention
  //     Input:  BxSxMxNxH (Packed QKV)
  //     Output: MxBxNxSxH
  //     qkv_add_bias: BxSxMxNxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  int in_offset = n * H + (m + s * M) * NH + b * NHS * M;
  const int out_offset = s * H + n * sequence_length * H + b * NHS + m * NHS * batch_size;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    if (nullptr != qkv_add_bias) {
      qkv_add_bias[in_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    }
    h += stride;
  }
}


template <typename T>
__global__ void AddBiasTransposeCutlass(const T* input, const T* biases, T* output, int v_head_size) {
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSx(NxH + NxH + NxH_v)  (Packed QKV where K and V has different hidden sizes)
  //     Output: BxNxSxH + BxNxSxH + BxNxSxH_v
  // B is batch_size, S is sequence_length, N is num_heads, H is qk_head_size, H_v is v_head_size
  int n = threadIdx.y;        // head_num_id
  int s = blockIdx.x;         // sequence_id
  int b = blockIdx.y;         // batch_id
  int m = blockIdx.z;         // matrix id (Q=0, K=1, V=2)
  const int h = threadIdx.x;  // head_element_id

  const int qk_head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int head_size = (m == 2 ? v_head_size : qk_head_size);

  const int total_head_size = num_heads * (qk_head_size + qk_head_size + v_head_size);

  int in_offset;
  int out_offset;
  int bias_offset;
  in_offset = b * (total_head_size * sequence_length) +  // B
              s * (total_head_size) +                    // S
              m * (qk_head_size * num_heads) +           // M
              n * head_size +                            // N
              h;                                         // H

  out_offset = m * (num_heads * qk_head_size * sequence_length * batch_size) +  // M
               b * (num_heads * head_size * sequence_length) +                  // B
               s * (num_heads * head_size) +                                    // S
               n * (head_size) +                                                // N
               h;                                                               // H

  bias_offset = m * (num_heads * qk_head_size) +  // M
                n * (head_size) +                 // N
                h;                                // H

  if (h < head_size) {
    output[out_offset] = input[in_offset] + biases[bias_offset];
  }
}

template <typename T>
__global__ void AddBiasUnpack(int M, const T* input, const T* biases, T* output) {
  // Format 4 to unpack TRT packed input format for memory efficient attention.
  //     Input:  BxSxNxMxH
  //     Output: MxBxSxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = m * head_size + n * M * H + (s * NH + b * NHS) * M;
  const int out_offset = n * head_size + s * NH + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    if (biases != nullptr) {
      output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    } else {
      output[out_offset + h] = input[in_offset + h];
    }
  }
}

template <typename T>
__global__ void AddBiasTransposeCutlass(int M, const T* input, const T* biases, T* output) {
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSxMxNxH
  //     Output: MxBxSxNxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * head_size + (m + s * M) * NH + b * NHS * M;
  const int out_offset = n * head_size + s * NH + b * NHS + m * NHS * batch_size;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeCutlassLarge(const int head_size, const T* input, const T* biases, T* output,
                                             const int M) {
  // Format 3 for cutlass memory efficient attention
  //     Input:  BxSxMxNxH (Packed QKV)
  //     Output: MxBxSxNxH
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;
  int in_offset = n * H + (m + s * M) * NH + b * NHS * M;
  const int out_offset = n * H + s * NH + b * NHS + m * NHS * batch_size;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
__global__ void AddBiasTranspose(const T* input, const T* biases, T* output) {
  // Format 0 for Separated Q, K, V (N*H <= 1024)
  //    Input:  MxBxSxNxH
  //    Output: MxBxNxSxH
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int head_size = blockDim.x;
  const int num_heads = blockDim.y;
  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * head_size;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = (s + n * sequence_length) * H + (b + m * batch_size) * NHS;

  const int h = threadIdx.x;
  if (h < head_size) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
  }
}

template <typename T>
__global__ void AddBiasTransposeLarge(const int head_size, const T* input, const T* biases, T* output) {
  // Format 0 for Separated Q, K, V (N*H > 1024)
  int n = threadIdx.y;
  int s = blockIdx.x;
  int b = blockIdx.y;
  int m = blockIdx.z;  // matrix id

  const int stride = blockDim.x;
  const int num_heads = blockDim.y;

  const int sequence_length = gridDim.x;
  const int batch_size = gridDim.y;

  const int H = head_size;
  const int NH = num_heads * H;
  const int NHS = NH * sequence_length;

  int in_offset = n * H + s * NH + (b + m * batch_size) * NHS;
  const int out_offset = (s + n * sequence_length) * H + (b + m * batch_size) * NHS;

  int h = threadIdx.x;
  while (h < H) {
    output[out_offset + h] = input[in_offset + h] + biases[m * NH + n * H + h];
    h += stride;
  }
}

template <typename T>
void InvokeAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const T* input, const T* biases, T* output, T* qkv_add_bias, const int v_head_size, int total_matrix_count,
    bool do_rotary = false, int original_past_sequence_length = 0) {
  assert(num_heads <= max_threads_per_block);

  if (do_rotary) {
    if (format != 1 && format != 3) {
      ORT_THROW("format must be 1 or 3 for rotary attention");
    }
    if (v_head_size != -1 && qk_head_size != v_head_size) {
      ORT_THROW("qk_head_size must be equal to v_head_size for rotary attention");
    }

    const int step = original_past_sequence_length == 0 ? sequence_length : original_past_sequence_length;
    size_t smem_size = 2 * qk_head_size * sizeof(T);

    const dim3 grid(sequence_length, num_heads, batch_size);
    const dim3 block((qk_head_size / 2 + 31) / 32 * 32, 1, 1);
    AddBiasTransposeQKV<T><<<grid, block, smem_size, stream>>>(total_matrix_count, input, biases, output,
                                                               qkv_add_bias, qk_head_size, qk_head_size,
                                                               step, format);
    return;
  }

  const dim3 grid(sequence_length, batch_size, num_matrices);
  if (qk_head_size * num_heads <= max_threads_per_block) {
    const dim3 block(qk_head_size, num_heads, 1);
    if (format == 2) {
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(input, biases, output);
    } else if (format == 1) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output, qkv_add_bias);
      } else {
        ORT_ENFORCE(total_matrix_count == 3);
        AddBiasTransposeQKV<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
      }
    } else if (format == 3) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeCutlass<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output);
      } else {
        ORT_ENFORCE(total_matrix_count == 3);
        AddBiasTransposeCutlass<T><<<grid, block, 0, stream>>>(input, biases, output, v_head_size);
      }
    } else if (format == 4) {  // format == 4
      AddBiasUnpack<T><<<grid, block, 0, stream>>>(total_matrix_count, input, biases, output);
    } else {  // format == 0
      AddBiasTranspose<T><<<grid, block, 0, stream>>>(input, biases, output);
    }
  } else {
    const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
    if (format == 2) {
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
    } else if (format == 1) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeQKVLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output,
                                                                qkv_add_bias, total_matrix_count);
      } else {
        // It is rare for hidden size > 4096 (for half precision) and qk_head_size != v_head_size.
        ORT_THROW("AddBiasTranspose (format 1) not implemented for hidden_size > max_threads_per_block when qk_head_size != v_head_size");
      }
    } else if (format == 3) {
      if (v_head_size == -1 || qk_head_size == v_head_size) {
        AddBiasTransposeCutlassLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output,
                                                                    total_matrix_count);
      } else {
        ORT_THROW("AddBiasTranspose (format 3) not implemented for hidden_size > max_threads_per_block when qk_head_size != v_head_size");
      }
    } else if (format == 4) {  // format == 4
      ORT_THROW("AddBiasTranspose (format 4) not implemented for hidden_size > max_threads_per_block");
    } else {  // format 0
      AddBiasTransposeLarge<T><<<grid, block, 0, stream>>>(qk_head_size, input, biases, output);
    }
  }
}

template <>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const half* input, const half* biases, half* output, bool enable_half4, const int v_head_size,
    half* qkv_add_bias, int total_matrix_count, bool do_rotary, int original_past_sequence_length) {
  total_matrix_count = std::max(num_matrices, total_matrix_count);
  if (enable_half4 && 0 == (qk_head_size % 4) && (v_head_size == -1 || 0 == (v_head_size % 4)) && !do_rotary) {
    const int H = qk_head_size / 4;
    const int H_v = v_head_size / 4;
    const Half4* input2 = reinterpret_cast<const Half4*>(input);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* output2 = reinterpret_cast<Half4*>(output);
    Half4* qkv_add_bias2 = reinterpret_cast<Half4*>(qkv_add_bias);
    InvokeAddBiasTranspose<Half4>(stream, num_matrices, format, max_threads_per_block,
                                  batch_size, sequence_length, num_heads, H, input2, biases2, output2,
                                  qkv_add_bias2, H_v, total_matrix_count);
  } else if (0 == (qk_head_size & 1) && (v_head_size == -1 || 0 == (v_head_size & 1)) && !do_rotary) {
    const int H = qk_head_size / 2;
    const int H_v = v_head_size / 2;
    const half2* input2 = reinterpret_cast<const half2*>(input);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* output2 = reinterpret_cast<half2*>(output);
    half2* qkv_add_bias2 = reinterpret_cast<half2*>(qkv_add_bias);
    InvokeAddBiasTranspose<half2>(stream, num_matrices, format, max_threads_per_block,
                                  batch_size, sequence_length, num_heads, H, input2, biases2, output2,
                                  qkv_add_bias2, H_v, total_matrix_count);
  } else {
    InvokeAddBiasTranspose<half>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size, input, biases, output,
        qkv_add_bias, v_head_size, total_matrix_count, do_rotary, original_past_sequence_length);
  }
}

template <>
void LaunchAddBiasTranspose(
    cudaStream_t stream, const int num_matrices, const int format, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int qk_head_size,
    const float* input, const float* biases, float* output, bool /*enable_half4*/,
    const int v_head_size, float* qkv_add_bias, int total_matrix_count, bool do_rotary,
    int original_past_sequence_length) {
  total_matrix_count = std::max(num_matrices, total_matrix_count);
  if (0 == (qk_head_size % 4) && (v_head_size == -1 || 0 == (v_head_size % 4)) && !do_rotary) {
    const int H = qk_head_size / 4;
    const float4* input2 = reinterpret_cast<const float4*>(input);
    const float4* biases2 = reinterpret_cast<const float4*>(biases);
    float4* output2 = reinterpret_cast<float4*>(output);
    float4* qkv_add_bias2 = reinterpret_cast<float4*>(qkv_add_bias);
    InvokeAddBiasTranspose<float4>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, H, input2, biases2, output2,
        qkv_add_bias2, v_head_size / 4, total_matrix_count);
  } else if (0 == (qk_head_size & 1) && (v_head_size == -1 || 0 == (v_head_size & 1)) && !do_rotary) {
    const int H = qk_head_size / 2;
    const float2* input2 = reinterpret_cast<const float2*>(input);
    const float2* biases2 = reinterpret_cast<const float2*>(biases);
    float2* output2 = reinterpret_cast<float2*>(output);
    float2* qkv_add_bias2 = reinterpret_cast<float2*>(qkv_add_bias);
    InvokeAddBiasTranspose<float2>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, H, input2, biases2, output2,
        qkv_add_bias2, v_head_size / 2, total_matrix_count);
  } else {
    InvokeAddBiasTranspose<float>(
        stream, num_matrices, format, max_threads_per_block,
        batch_size, sequence_length, num_heads, qk_head_size, input, biases, output,
        qkv_add_bias, v_head_size, total_matrix_count, do_rotary, original_past_sequence_length);
  }
}

template <typename T>
void InvokeAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int num_heads, const int head_size,
    const T* biases, const T* query, const T* key, const T* value, T* output,
    bool is_cross_attention, int kv_sequence_length) {
  if (!is_cross_attention) {
    ORT_ENFORCE(sequence_length == kv_sequence_length);
    constexpr int num_matrices = 3;
    const dim3 grid(sequence_length, batch_size, num_matrices);
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, key, value, biases, output);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, key, value, biases, output);
    }
  } else {  // cross attention
    // Q: add bias
    {
      constexpr int num_matrices = 1;
      const dim3 grid(sequence_length, batch_size, num_matrices);
      if (head_size * num_heads <= max_threads_per_block) {
        const dim3 block(head_size, num_heads, 1);
        AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, biases, output);
      } else {
        const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
        AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, biases, output);
      }
    }

    // KV: add bias and pack kv
    {
      constexpr int num_matrices = 2;
      const dim3 grid(kv_sequence_length, batch_size, num_matrices);
      T* packed_kv = output + batch_size * sequence_length * num_heads * head_size;
      if (head_size * num_heads <= max_threads_per_block) {
        const dim3 block(head_size, num_heads, 1);
        AddBiasTransposeTrtKV<T><<<grid, block, 0, stream>>>(key, value, biases, packed_kv);
      } else {
        const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
        AddBiasTransposeTrtKVLarge<T><<<grid, block, 0, stream>>>(head_size, key, value, biases, packed_kv);
      }
    }
  }
}

template <>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const float* biases, const float* query, const float* key, const float* value, float* output,
    bool is_cross_attention, int kv_sequence_length) {
  ORT_ENFORCE(false, "Shall not call this since fused kernel does not support float input.");
}

template <>
void LaunchAddBiasTransposeTrt(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length,
    const int num_heads, const int head_size,
    const half* biases, const half* query, const half* key, const half* value, half* output,
    bool is_cross_attention, int kv_sequence_length) {
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    const Half4* query2 = reinterpret_cast<const Half4*>(query);
    const Half4* key2 = reinterpret_cast<const Half4*>(key);
    const Half4* value2 = reinterpret_cast<const Half4*>(value);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* output2 = reinterpret_cast<Half4*>(output);
    InvokeAddBiasTransposeTrt<Half4>(stream, max_threads_per_block,
                                     batch_size, sequence_length, num_heads, H,
                                     biases2, query2, key2, value2, output2, is_cross_attention, kv_sequence_length);
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    const half2* query2 = reinterpret_cast<const half2*>(query);
    const half2* key2 = reinterpret_cast<const half2*>(key);
    const half2* value2 = reinterpret_cast<const half2*>(value);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* output2 = reinterpret_cast<half2*>(output);
    InvokeAddBiasTransposeTrt<half2>(stream, max_threads_per_block,
                                     batch_size, sequence_length, num_heads, H,
                                     biases2, query2, key2, value2, output2, is_cross_attention, kv_sequence_length);
  } else {
    InvokeAddBiasTransposeTrt<half>(stream, max_threads_per_block,
                                    batch_size, sequence_length, num_heads, head_size,
                                    biases, query, key, value, output, is_cross_attention, kv_sequence_length);
  }
}

template <typename T>
void InvokeAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const T* biases, const T* query, const T* key, const T* value, T* q, T* k, T* v) {
  assert(num_heads <= max_threads_per_block);
  constexpr int num_matrices = 1;
  // Q
  {
    const dim3 grid(sequence_length, batch_size, num_matrices);
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(query, biases, q);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, query, biases, q);
    }
  }
  // K
  {
    const dim3 grid(kv_sequence_length, batch_size, num_matrices);
    const T* biases_k = biases + num_heads * head_size;

    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(key, biases_k, k);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(head_size, key, biases_k, k);
    }
  }

  // V
  {
    const dim3 grid(kv_sequence_length, batch_size, num_matrices);

    const T* biases_v = biases + 2 * num_heads * head_size;
    if (v_head_size * num_heads <= max_threads_per_block) {
      const dim3 block(v_head_size, num_heads, 1);
      AddBiasTransposeTrt<T><<<grid, block, 0, stream>>>(value, biases_v, v);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      AddBiasTransposeTrtLarge<T><<<grid, block, 0, stream>>>(v_head_size, value, biases_v, v);
    }
  }
}

template <>
void LaunchAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const float* biases, const float* query, const float* key, const float* value, float* q, float* k, float* v) {
  if (0 == (head_size % 4) && 0 == (v_head_size % 4)) {
    const int H = head_size / 4;
    const int H_v = v_head_size / 4;
    const float4* query2 = reinterpret_cast<const float4*>(query);
    const float4* key2 = reinterpret_cast<const float4*>(key);
    const float4* value2 = reinterpret_cast<const float4*>(value);
    const float4* biases2 = reinterpret_cast<const float4*>(biases);
    float4* q2 = reinterpret_cast<float4*>(q);
    float4* k2 = reinterpret_cast<float4*>(k);
    float4* v2 = reinterpret_cast<float4*>(v);
    InvokeAddBias<float4>(stream, max_threads_per_block,
                          batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                          biases2, query2, key2, value2, q2, k2, v2);
  } else if (0 == (head_size & 1) && 0 == (v_head_size & 1)) {
    const int H = head_size / 2;
    const int H_v = v_head_size / 2;
    const float2* query2 = reinterpret_cast<const float2*>(query);
    const float2* key2 = reinterpret_cast<const float2*>(key);
    const float2* value2 = reinterpret_cast<const float2*>(value);
    const float2* biases2 = reinterpret_cast<const float2*>(biases);
    float2* q2 = reinterpret_cast<float2*>(q);
    float2* k2 = reinterpret_cast<float2*>(k);
    float2* v2 = reinterpret_cast<float2*>(v);
    InvokeAddBias<float2>(stream, max_threads_per_block,
                          batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                          biases2, query2, key2, value2, q2, k2, v2);
  } else {
    InvokeAddBias<float>(stream, max_threads_per_block,
                         batch_size, sequence_length, kv_sequence_length, num_heads, head_size, v_head_size,
                         biases, query, key, value, q, k, v);
  }
}

template <>
void LaunchAddBias(
    cudaStream_t stream, const int max_threads_per_block,
    const int batch_size, const int sequence_length, const int kv_sequence_length,
    const int num_heads, const int head_size, const int v_head_size,
    const half* biases, const half* query, const half* key, const half* value, half* q, half* k, half* v) {
  if (0 == (head_size % 4) && 0 == (v_head_size % 4)) {
    const int H = head_size / 4;
    const int H_v = v_head_size / 4;
    const Half4* query2 = reinterpret_cast<const Half4*>(query);
    const Half4* key2 = reinterpret_cast<const Half4*>(key);
    const Half4* value2 = reinterpret_cast<const Half4*>(value);
    const Half4* biases2 = reinterpret_cast<const Half4*>(biases);
    Half4* q2 = reinterpret_cast<Half4*>(q);
    Half4* k2 = reinterpret_cast<Half4*>(k);
    Half4* v2 = reinterpret_cast<Half4*>(v);
    InvokeAddBias<Half4>(stream, max_threads_per_block,
                         batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                         biases2, query2, key2, value2, q2, k2, v2);
  } else if (0 == (head_size & 1) && 0 == (v_head_size & 1)) {
    const int H = head_size / 2;
    const int H_v = v_head_size / 2;
    const half2* query2 = reinterpret_cast<const half2*>(query);
    const half2* key2 = reinterpret_cast<const half2*>(key);
    const half2* value2 = reinterpret_cast<const half2*>(value);
    const half2* biases2 = reinterpret_cast<const half2*>(biases);
    half2* q2 = reinterpret_cast<half2*>(q);
    half2* k2 = reinterpret_cast<half2*>(k);
    half2* v2 = reinterpret_cast<half2*>(v);
    InvokeAddBias<half2>(stream, max_threads_per_block,
                         batch_size, sequence_length, kv_sequence_length, num_heads, H, H_v,
                         biases2, query2, key2, value2, q2, k2, v2);
  } else {
    InvokeAddBias<half>(stream, max_threads_per_block,
                        batch_size, sequence_length, kv_sequence_length, num_heads, head_size, v_head_size,
                        biases, query, key, value, q, k, v);
  }
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
