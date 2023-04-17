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

// vector with zero lanes
NPY_FINLINE npyv_u8
npyv_zero_u8()
{
    return svdup_n_u8(0);
}
NPY_FINLINE npyv_u16
npyv_zero_u16()
{
    return svdup_n_u16(0);
}
NPY_FINLINE npyv_u32
npyv_zero_u32()
{
    return svdup_n_u32(0);
}
NPY_FINLINE npyv_u64
npyv_zero_u64()
{
    return svdup_n_u64(0);
}
NPY_FINLINE npyv_s8
npyv_zero_s8()
{
    return svdup_n_s8(0);
}
NPY_FINLINE npyv_s16
npyv_zero_s16()
{
    return svdup_n_s16(0);
}
NPY_FINLINE npyv_s32
npyv_zero_s32()
{
    return svdup_n_s32(0);
}
NPY_FINLINE npyv_s64
npyv_zero_s64()
{
    return svdup_n_s64(0);
}
NPY_FINLINE npyv_f32
npyv_zero_f32()
{
    return svdup_n_f32(0.0f);
}
NPY_FINLINE npyv_f64
npyv_zero_f64()
{
    return svdup_n_f64(0.0);
}

// vector with a specific value set to all lanes
#define npyv_setall_u8 svdup_n_u8
#define npyv_setall_s8 svdup_n_s8
#define npyv_setall_u16 svdup_n_u16
#define npyv_setall_s16 svdup_n_s16
#define npyv_setall_u32 svdup_n_u32
#define npyv_setall_s32 svdup_n_s32
#define npyv_setall_u64 svdup_n_u64
#define npyv_setall_s64 svdup_n_s64
#define npyv_setall_f32 svdup_n_f32
#define npyv_setall_f64 svdup_n_f64

/**
 * vector with specific values set to each lane and
 * set a specific value to all remained lanes
 *
 */

// Per lane select
#define npyv_select_u8 svsel_u8
#define npyv_select_s8 svsel_s8
#define npyv_select_u16 svsel_u16
#define npyv_select_s16 svsel_s16
#define npyv_select_u32 svsel_u32
#define npyv_select_s32 svsel_s32
#define npyv_select_u64 svsel_u64
#define npyv_select_s64 svsel_s64
#define npyv_select_f32 svsel_f32
#define npyv_select_f64 svsel_f64

// Reinterpret
#define npyv_reinterpret_u8_u8(X) X
#define npyv_reinterpret_u8_s8 svreinterpret_u8_s8
#define npyv_reinterpret_u8_u16 svreinterpret_u8_u16
#define npyv_reinterpret_u8_s16 svreinterpret_u8_s16
#define npyv_reinterpret_u8_u32 svreinterpret_u8_u32
#define npyv_reinterpret_u8_s32 svreinterpret_u8_s32
#define npyv_reinterpret_u8_u64 svreinterpret_u8_u64
#define npyv_reinterpret_u8_s64 svreinterpret_u8_s64
#define npyv_reinterpret_u8_f32 svreinterpret_u8_f32
#define npyv_reinterpret_u8_f64 svreinterpret_u8_f64

#define npyv_reinterpret_s8_s8(X) X
#define npyv_reinterpret_s8_u8 svreinterpret_s8_u8
#define npyv_reinterpret_s8_u16 svreinterpret_s8_u16
#define npyv_reinterpret_s8_s16 svreinterpret_s8_s16
#define npyv_reinterpret_s8_u32 svreinterpret_s8_u32
#define npyv_reinterpret_s8_s32 svreinterpret_s8_s32
#define npyv_reinterpret_s8_u64 svreinterpret_s8_u64
#define npyv_reinterpret_s8_s64 svreinterpret_s8_s64
#define npyv_reinterpret_s8_f32 svreinterpret_s8_f32
#define npyv_reinterpret_s8_f64 svreinterpret_s8_f64

#define npyv_reinterpret_u16_u16(X) X
#define npyv_reinterpret_u16_u8 svreinterpret_u16_u8
#define npyv_reinterpret_u16_s8 svreinterpret_u16_s8
#define npyv_reinterpret_u16_s16 svreinterpret_u16_s16
#define npyv_reinterpret_u16_u32 svreinterpret_u16_u32
#define npyv_reinterpret_u16_s32 svreinterpret_u16_s32
#define npyv_reinterpret_u16_u64 svreinterpret_u16_u64
#define npyv_reinterpret_u16_s64 svreinterpret_u16_s64
#define npyv_reinterpret_u16_f32 svreinterpret_u16_f32
#define npyv_reinterpret_u16_f64 svreinterpret_u16_f64

#define npyv_reinterpret_s16_s16(X) X
#define npyv_reinterpret_s16_u8 svreinterpret_s16_u8
#define npyv_reinterpret_s16_s8 svreinterpret_s16_s8
#define npyv_reinterpret_s16_u16 svreinterpret_s16_u16
#define npyv_reinterpret_s16_u32 svreinterpret_s16_u32
#define npyv_reinterpret_s16_s32 svreinterpret_s16_s32
#define npyv_reinterpret_s16_u64 svreinterpret_s16_u64
#define npyv_reinterpret_s16_s64 svreinterpret_s16_s64
#define npyv_reinterpret_s16_f32 svreinterpret_s16_f32
#define npyv_reinterpret_s16_f64 svreinterpret_s16_f64

#define npyv_reinterpret_u32_u32(X) X
#define npyv_reinterpret_u32_u8 svreinterpret_u32_u8
#define npyv_reinterpret_u32_s8 svreinterpret_u32_s8
#define npyv_reinterpret_u32_u16 svreinterpret_u32_u16
#define npyv_reinterpret_u32_s16 svreinterpret_u32_s16
#define npyv_reinterpret_u32_s32 svreinterpret_u32_s32
#define npyv_reinterpret_u32_u64 svreinterpret_u32_u64
#define npyv_reinterpret_u32_s64 svreinterpret_u32_s64
#define npyv_reinterpret_u32_f32 svreinterpret_u32_f32
#define npyv_reinterpret_u32_f64 svreinterpret_u32_f64

#define npyv_reinterpret_s32_s32(X) X
#define npyv_reinterpret_s32_u8 svreinterpret_s32_u8
#define npyv_reinterpret_s32_s8 svreinterpret_s32_s8
#define npyv_reinterpret_s32_u16 svreinterpret_s32_u16
#define npyv_reinterpret_s32_s16 svreinterpret_s32_s16
#define npyv_reinterpret_s32_u32 svreinterpret_s32_u32
#define npyv_reinterpret_s32_u64 svreinterpret_s32_u64
#define npyv_reinterpret_s32_s64 svreinterpret_s32_s64
#define npyv_reinterpret_s32_f32 svreinterpret_s32_f32
#define npyv_reinterpret_s32_f64 svreinterpret_s32_f64

#define npyv_reinterpret_u64_u64(X) X
#define npyv_reinterpret_u64_u8 svreinterpret_u64_u8
#define npyv_reinterpret_u64_s8 svreinterpret_u64_s8
#define npyv_reinterpret_u64_u16 svreinterpret_u64_u16
#define npyv_reinterpret_u64_s16 svreinterpret_u64_s16
#define npyv_reinterpret_u64_u32 svreinterpret_u64_u32
#define npyv_reinterpret_u64_s32 svreinterpret_u64_s32
#define npyv_reinterpret_u64_s64 svreinterpret_u64_s64
#define npyv_reinterpret_u64_f32 svreinterpret_u64_f32
#define npyv_reinterpret_u64_f64 svreinterpret_u64_f64

#define npyv_reinterpret_s64_s64(X) X
#define npyv_reinterpret_s64_u8 svreinterpret_s64_u8
#define npyv_reinterpret_s64_s8 svreinterpret_s64_s8
#define npyv_reinterpret_s64_u16 svreinterpret_s64_u16
#define npyv_reinterpret_s64_s16 svreinterpret_s64_s16
#define npyv_reinterpret_s64_u32 svreinterpret_s64_u32
#define npyv_reinterpret_s64_s32 svreinterpret_s64_s32
#define npyv_reinterpret_s64_u64 svreinterpret_s64_u64
#define npyv_reinterpret_s64_f32 svreinterpret_s64_f32
#define npyv_reinterpret_s64_f64 svreinterpret_s64_f64

#define npyv_reinterpret_f32_f32(X) X
#define npyv_reinterpret_f32_u8 svreinterpret_f32_u8
#define npyv_reinterpret_f32_s8 svreinterpret_f32_s8
#define npyv_reinterpret_f32_u16 svreinterpret_f32_u16
#define npyv_reinterpret_f32_s16 svreinterpret_f32_s16
#define npyv_reinterpret_f32_u32 svreinterpret_f32_u32
#define npyv_reinterpret_f32_s32 svreinterpret_f32_s32
#define npyv_reinterpret_f32_u64 svreinterpret_f32_u64
#define npyv_reinterpret_f32_s64 svreinterpret_f32_s64
#define npyv_reinterpret_f32_f64 svreinterpret_f32_f64

#define npyv_reinterpret_f64_f64(X) X
#define npyv_reinterpret_f64_u8 svreinterpret_f64_u8
#define npyv_reinterpret_f64_s8 svreinterpret_f64_s8
#define npyv_reinterpret_f64_u16 svreinterpret_f64_u16
#define npyv_reinterpret_f64_s16 svreinterpret_f64_s16
#define npyv_reinterpret_f64_u32 svreinterpret_f64_u32
#define npyv_reinterpret_f64_s32 svreinterpret_f64_s32
#define npyv_reinterpret_f64_u64 svreinterpret_f64_u64
#define npyv_reinterpret_f64_s64 svreinterpret_f64_s64
#define npyv_reinterpret_f64_f32 svreinterpret_f64_f32

#define npyv_cleanup() ((void)0)

//#endif  // _NPY_SIMD_NEON_MISC_H
