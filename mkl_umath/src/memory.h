/*
 * Copyright (c) 2005-2023, NumPy Developers.
 * Copyright 2023 FUJITSU LIMITED
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of Intel Corporation nor the names of its contributors
 *       may be used to endorse or promote products derived from this software
 *       without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#define NPY_SIMD_WIDTH (svcntb() * 8) // 実行時に取得するように変更
#define NPY_SIMD_MAXLOAD_STRIDE32 (0x7fffffff / 16)
#define NPY_SIMD_MAXSTORE_STRIDE32 (0x7fffffff / 16)

typedef svuint8_t npyv_u8;
typedef svint8_t npyv_s8;
typedef svuint16_t npyv_u16;
typedef svint16_t npyv_s16;
typedef svuint32_t npyv_u32;
typedef svint32_t npyv_s32;
typedef svuint64_t npyv_u64;
typedef svint64_t npyv_s64;
typedef svfloat32_t npyv_f32;
typedef svfloat64_t npyv_f64;

// lane type by intrin suffix
typedef npy_uint8  npyv_lanetype_u8;
typedef npy_int8   npyv_lanetype_s8;
typedef npy_uint16 npyv_lanetype_u16;
typedef npy_int16  npyv_lanetype_s16;
typedef npy_uint32 npyv_lanetype_u32;
typedef npy_int32  npyv_lanetype_s32;
typedef npy_uint64 npyv_lanetype_u64;
typedef npy_int64  npyv_lanetype_s64;
typedef float      npyv_lanetype_f32;
typedef double     npyv_lanetype_f64;


#include "misc.h"

/*********************************
 * Partial Load
 *********************************/
//// 32
NPY_FINLINE svint32_t
npyv_load_till_s32(const npy_int32 *ptr, npy_uintp nlane, npy_int32 fill)
{
    assert(nlane > 0);
    if (nlane == NPY_SIMD_WIDTH / sizeof(int32_t)) {
        return svld1(svptrue_b32(), ptr);
    }
    else {
        const svint32_t vfill = svdup_s32(fill);
        const svint32_t idx = svindex_s32(0, 1);
        const svbool_t mask = svcmplt_n_s32(svptrue_b32(), idx, nlane);
        return svsel_s32(mask, svld1_s32(mask, ptr), vfill);
    }
}
// fill zero to rest lanes
NPY_FINLINE svint32_t
npyv_load_tillz_s32(const npy_int32 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == NPY_SIMD_WIDTH / sizeof(int32_t)) {
        return svld1(svptrue_b32(), ptr);
    }
    else {
        const svint32_t idx = svindex_s32(0, 1);
        const svbool_t mask = svcmplt_n_s32(svptrue_b32(), idx, nlane);
        return svld1_s32(mask, ptr);
    }
}
//// 64
NPY_FINLINE svint64_t
npyv_load_till_s64(const npy_int64 *ptr, npy_uintp nlane, npy_int64 fill)
{
    assert(nlane > 0);
    if (nlane == NPY_SIMD_WIDTH / sizeof(int64_t)) {
        return svld1(svptrue_b64(), ptr);
    }
    else {
        const svint64_t vfill = svdup_s64(fill);
        const svint64_t idx = svindex_s64(0, 1);
        const svbool_t mask = svcmplt_n_s64(svptrue_b64(), idx, nlane);
        return svsel_s64(mask, svld1_s64(mask, ptr), vfill);
    }
}
// fill zero to rest lanes
NPY_FINLINE svint64_t
npyv_load_tillz_s64(const npy_int64 *ptr, npy_uintp nlane)
{
    assert(nlane > 0);
    if (nlane == NPY_SIMD_WIDTH / sizeof(int64_t)) {
        return svld1(svptrue_b64(), ptr);
    }
    else {
        const svint64_t idx = svindex_s64(0, 1);
        const svbool_t mask = svcmplt_n_s64(svptrue_b64(), idx, nlane);
        return svld1_s64(mask, ptr);
    }
}

/*********************************
 * Non-contiguous partial load
 *********************************/
//// 32
NPY_FINLINE svint32_t
npyv_loadn_till_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                    npy_int32 fill)
{
    assert(nlane > 0);
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const svint32_t steps = svindex_s32(0, 1);
    const svint32_t idx =
            svmul_s32_x(svptrue_b32(), steps, svdup_s32((int)stride));

    if (nlane == NPY_SIMD_WIDTH / sizeof(int32_t)) {
        return svld1_gather_s32offset_s32(svptrue_b32(), ptr, idx);
    }
    else {
        const svint32_t tmp = svindex_s32(0, 1);
        const svbool_t mask = svcmplt_n_s32(svptrue_b64(), tmp, nlane);
        const svint32_t vfill = svdup_s32(fill);
        return svsel_s32(mask, svld1_gather_s32index_s32(mask, ptr, idx),
                         vfill);
    }
}
// fill zero to rest lanes
NPY_FINLINE svint32_t
npyv_loadn_tillz_s32(const npy_int32 *ptr, npy_intp stride, npy_uintp nlane)
{
    return npyv_loadn_till_s32(ptr, stride, nlane, 0);
}
//// 64
NPY_FINLINE svint64_t
npyv_loadn_till_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                    npy_int64 fill)
{
    assert(nlane > 0);
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const svint64_t steps = svindex_s64(0, 1);
    const svint64_t idx =
            svmul_s64_x(svptrue_b64(), steps, svdup_s64((int)stride));

    if (nlane == NPY_SIMD_WIDTH / sizeof(int64_t)) {
        return svld1_gather_s64offset_s64(svptrue_b64(), ptr, idx);
    }
    else {
        const svint64_t tmp = svindex_s64(0, 1);
        const svbool_t mask = svcmplt_n_s64(svptrue_b64(), tmp, nlane);
        const svint64_t vfill = svdup_s64(fill);
        return svsel_s64(mask, svld1_gather_s64index_s64(mask, ptr, idx),
                         vfill);
    }
}
// fill zero to rest lanes
NPY_FINLINE svint64_t
npyv_loadn_tillz_s64(const npy_int64 *ptr, npy_intp stride, npy_uintp nlane)
{
    return npyv_loadn_till_s64(ptr, stride, nlane, 0);
}
/*********************************
 * Partial store
 *********************************/
//// 32
NPY_FINLINE void
npyv_store_till_s32(npy_int32 *ptr, npy_uintp nlane, npyv_s32 a)
{
    assert(nlane > 0);
    if (nlane == NPY_SIMD_WIDTH / sizeof(int32_t)) {
        svst1_s32(svptrue_b32(), ptr, a);
    }
    else {
        const svint32_t idx = svindex_s32(0, 1);
        const svbool_t mask = svcmplt_n_s32(svptrue_b32(), idx, nlane);
        svst1_s32(mask, ptr, a);
    }
}
//// 64
NPY_FINLINE void
npyv_store_till_s64(npy_int64 *ptr, npy_uintp nlane, npyv_s64 a)
{
    assert(nlane > 0);
    if (nlane == NPY_SIMD_WIDTH / sizeof(int64_t)) {
        svst1_s64(svptrue_b64(), ptr, a);
    }
    else {
        const svint64_t idx = svindex_s64(0, 1);
        const svbool_t mask = svcmplt_n_s64(svptrue_b64(), idx, nlane);
        svst1_s64(mask, ptr, a);
    }
}
/*********************************
 * Non-contiguous partial store
 *********************************/
//// 32
NPY_FINLINE void
npyv_storen_till_s32(npy_int32 *ptr, npy_intp stride, npy_uintp nlane,
                     npyv_s32 a)
{
    assert(nlane > 0);
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const svint32_t steps = svindex_s32(0, 1);
    const svint32_t idx =
            svmul_s32_x(svptrue_b32(), steps, svdup_s32((int)stride));

    if (nlane == NPY_SIMD_WIDTH / sizeof(int32_t)) {
        svst1_scatter_s32offset_s32(svptrue_b32(), ptr, idx, a);
    }
    else {
        const svint32_t tmp = svindex_s32(0, 1);
        const svbool_t mask = svcmplt_n_s32(svptrue_b64(), tmp, nlane);
        svst1_scatter_s32offset_s32(mask, ptr, idx, a);
    }
}
//// 64
NPY_FINLINE void
npyv_storen_till_s64(npy_int64 *ptr, npy_intp stride, npy_uintp nlane,
                     npyv_s64 a)
{
    assert(nlane > 0);
    assert(llabs(stride) <= NPY_SIMD_MAXLOAD_STRIDE32);
    const svint64_t steps = svindex_s64(0, 1);
    const svint64_t idx =
            svmul_s64_x(svptrue_b64(), steps, svdup_s64((int)stride));

    if (nlane == NPY_SIMD_WIDTH / sizeof(int64_t)) {
        svst1_scatter_s64offset_s64(svptrue_b64(), ptr, idx, a);
    }
    else {
        const svint64_t tmp = svindex_s64(0, 1);
        const svbool_t mask = svcmplt_n_s64(svptrue_b64(), tmp, nlane);
        svst1_scatter_s64offset_s64(mask, ptr, idx, a);
    }
}

/*****************************************************************************
 * Implement partial load/store for u32/f32/u64/f64... via reinterpret cast
 *****************************************************************************/
#define NPYV_IMPL_SVE_REST_PARTIAL_TYPES(F_SFX, T_SFX)                        \
    NPY_FINLINE npyv_##F_SFX npyv_load_till_##F_SFX(                          \
            const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane,                \
            npyv_lanetype_##F_SFX fill)                                       \
    {                                                                         \
        union {                                                               \
            npyv_lanetype_##F_SFX from_##F_SFX;                               \
            npyv_lanetype_##T_SFX to_##T_SFX;                                 \
        } pun = {.from_##F_SFX = fill};                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_till_##T_SFX(     \
                (const npyv_lanetype_##T_SFX *)ptr, nlane, pun.to_##T_SFX));  \
    }                                                                         \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_till_##F_SFX(                         \
            const npyv_lanetype_##F_SFX *ptr, npy_intp stride,                \
            npy_uintp nlane, npyv_lanetype_##F_SFX fill)                      \
    {                                                                         \
        union {                                                               \
            npyv_lanetype_##F_SFX from_##F_SFX;                               \
            npyv_lanetype_##T_SFX to_##T_SFX;                                 \
        } pun = {.from_##F_SFX = fill};                                       \
        return npyv_reinterpret_##F_SFX##_##T_SFX(                            \
                npyv_loadn_till_##T_SFX((const npyv_lanetype_##T_SFX *)ptr,   \
                                        stride, nlane, pun.to_##T_SFX));      \
    }                                                                         \
    NPY_FINLINE npyv_##F_SFX npyv_load_tillz_##F_SFX(                         \
            const npyv_lanetype_##F_SFX *ptr, npy_uintp nlane)                \
    {                                                                         \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_load_tillz_##T_SFX(    \
                (const npyv_lanetype_##T_SFX *)ptr, nlane));                  \
    }                                                                         \
    NPY_FINLINE npyv_##F_SFX npyv_loadn_tillz_##F_SFX(                        \
            const npyv_lanetype_##F_SFX *ptr, npy_intp stride,                \
            npy_uintp nlane)                                                  \
    {                                                                         \
        return npyv_reinterpret_##F_SFX##_##T_SFX(npyv_loadn_tillz_##T_SFX(   \
                (const npyv_lanetype_##T_SFX *)ptr, stride, nlane));          \
    }                                                                         \
    NPY_FINLINE void npyv_store_till_##F_SFX(npyv_lanetype_##F_SFX *ptr,      \
                                             npy_uintp nlane, npyv_##F_SFX a) \
    {                                                                         \
        npyv_store_till_##T_SFX((npyv_lanetype_##T_SFX *)ptr, nlane,          \
                                npyv_reinterpret_##T_SFX##_##F_SFX(a));       \
    }                                                                         \
    NPY_FINLINE void npyv_storen_till_##F_SFX(                                \
            npyv_lanetype_##F_SFX *ptr, npy_intp stride, npy_uintp nlane,     \
            npyv_##F_SFX a)                                                   \
    {                                                                         \
        npyv_storen_till_##T_SFX((npyv_lanetype_##T_SFX *)ptr, stride, nlane, \
                                 npyv_reinterpret_##T_SFX##_##F_SFX(a));      \
    }

NPYV_IMPL_SVE_REST_PARTIAL_TYPES(u32, s32)
NPYV_IMPL_SVE_REST_PARTIAL_TYPES(f32, s32)
NPYV_IMPL_SVE_REST_PARTIAL_TYPES(u64, s64)
NPYV_IMPL_SVE_REST_PARTIAL_TYPES(f64, s64)

//#endif  // _NPY_SIMD_SVE_MEMORY_H
