// Copyright (c) Team CharLS.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// SIMD utilities for CharLS
// Provides vectorized operations for line buffer processing

#include <cstddef>
#include <cstdint>
#include <cstring>

// Detect SIMD support at compile time
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CHARLS_X86_SIMD_AVAILABLE 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CHARLS_ARM_NEON_AVAILABLE 1
#include <arm_neon.h>
#endif

// Disable cast-align warnings for this file.
// SIMD intrinsics like _mm_loadu_si128 are explicitly designed for unaligned access,
// but GCC still warns about the pointer cast alignment.
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcast-align"
#endif

namespace charls {
namespace simd {

// Runtime CPU feature detection for x86
#ifdef CHARLS_X86_SIMD_AVAILABLE

inline bool cpu_supports_sse2() noexcept
{
#ifdef __SSE2__
    return true; // Compile-time guarantee
#else
    // Runtime detection would go here
    return false;
#endif
}

inline bool cpu_supports_avx2() noexcept
{
#ifdef __AVX2__
    return true; // Compile-time guarantee
#else
    return false;
#endif
}

/// <summary>
/// SIMD-optimized masked copy for 8-bit samples using SSE2.
/// Processes 16 bytes at a time.
/// </summary>
inline void copy_samples_masked_sse2(const uint8_t* source, uint8_t* destination,
                                      size_t byte_count, uint8_t mask) noexcept
{
    const __m128i mask_vec = _mm_set1_epi8(static_cast<char>(mask));
    
    size_t i = 0;
    // Process 16 bytes per iteration
    const size_t simd_end = byte_count - (byte_count % 16);
    for (; i < simd_end; i += 16)
    {
        const __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const __m128i result = _mm_and_si128(data, mask_vec);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + i), result);
    }
    
    // Scalar fallback for remainder
    for (; i < byte_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
}

/// <summary>
/// SIMD-optimized masked copy for 16-bit samples using SSE2.
/// Processes 8 samples (16 bytes) at a time.
/// </summary>
inline void copy_samples_masked_16bit_sse2(const uint16_t* source, uint16_t* destination,
                                            size_t sample_count, uint16_t mask) noexcept
{
    const __m128i mask_vec = _mm_set1_epi16(static_cast<short>(mask));
    
    size_t i = 0;
    // Process 8 samples per iteration
    const size_t simd_end = sample_count - (sample_count % 8);
    for (; i < simd_end; i += 8)
    {
        const __m128i data = _mm_loadu_si128(reinterpret_cast<const __m128i*>(source + i));
        const __m128i result = _mm_and_si128(data, mask_vec);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(destination + i), result);
    }
    
    // Scalar fallback for remainder
    for (; i < sample_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
}

#ifdef __AVX2__
/// <summary>
/// SIMD-optimized masked copy for 8-bit samples using AVX2.
/// Processes 32 bytes at a time.
/// </summary>
inline void copy_samples_masked_avx2(const uint8_t* source, uint8_t* destination,
                                      size_t byte_count, uint8_t mask) noexcept
{
    const __m256i mask_vec = _mm256_set1_epi8(static_cast<char>(mask));
    
    size_t i = 0;
    // Process 32 bytes per iteration
    const size_t simd_end = byte_count - (byte_count % 32);
    for (; i < simd_end; i += 32)
    {
        const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + i));
        const __m256i result = _mm256_and_si256(data, mask_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + i), result);
    }
    
    // Handle remaining bytes with SSE2
    copy_samples_masked_sse2(source + i, destination + i, byte_count - i, mask);
}

/// <summary>
/// SIMD-optimized masked copy for 16-bit samples using AVX2.
/// Processes 16 samples (32 bytes) at a time.
/// </summary>
inline void copy_samples_masked_16bit_avx2(const uint16_t* source, uint16_t* destination,
                                            size_t sample_count, uint16_t mask) noexcept
{
    const __m256i mask_vec = _mm256_set1_epi16(static_cast<short>(mask));
    
    size_t i = 0;
    // Process 16 samples per iteration
    const size_t simd_end = sample_count - (sample_count % 16);
    for (; i < simd_end; i += 16)
    {
        const __m256i data = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(source + i));
        const __m256i result = _mm256_and_si256(data, mask_vec);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(destination + i), result);
    }
    
    // Handle remaining samples with SSE2
    copy_samples_masked_16bit_sse2(source + i, destination + i, sample_count - i, mask);
}
#endif // __AVX2__

#endif // CHARLS_X86_SIMD_AVAILABLE

// ARM NEON implementations
#ifdef CHARLS_ARM_NEON_AVAILABLE

/// <summary>
/// SIMD-optimized masked copy for 8-bit samples using NEON.
/// Processes 16 bytes at a time.
/// </summary>
inline void copy_samples_masked_neon(const uint8_t* source, uint8_t* destination,
                                      size_t byte_count, uint8_t mask) noexcept
{
    const uint8x16_t mask_vec = vdupq_n_u8(mask);
    
    size_t i = 0;
    // Process 16 bytes per iteration
    const size_t simd_end = byte_count - (byte_count % 16);
    for (; i < simd_end; i += 16)
    {
        const uint8x16_t data = vld1q_u8(source + i);
        const uint8x16_t result = vandq_u8(data, mask_vec);
        vst1q_u8(destination + i, result);
    }
    
    // Scalar fallback for remainder
    for (; i < byte_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
}

/// <summary>
/// SIMD-optimized masked copy for 16-bit samples using NEON.
/// Processes 8 samples (16 bytes) at a time.
/// </summary>
inline void copy_samples_masked_16bit_neon(const uint16_t* source, uint16_t* destination,
                                            size_t sample_count, uint16_t mask) noexcept
{
    const uint16x8_t mask_vec = vdupq_n_u16(mask);
    
    size_t i = 0;
    // Process 8 samples per iteration
    const size_t simd_end = sample_count - (sample_count % 8);
    for (; i < simd_end; i += 8)
    {
        const uint16x8_t data = vld1q_u16(source + i);
        const uint16x8_t result = vandq_u16(data, mask_vec);
        vst1q_u16(destination + i, result);
    }
    
    // Scalar fallback for remainder
    for (; i < sample_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
}

#endif // CHARLS_ARM_NEON_AVAILABLE

/// <summary>
/// Dispatcher function that selects the best SIMD implementation at compile time.
/// Falls back to scalar if no SIMD available.
/// </summary>
template<typename SampleType>
inline void copy_samples_masked_simd(const SampleType* source, SampleType* destination,
                                      size_t sample_count, SampleType mask) noexcept
{
    // Minimum threshold for SIMD to be beneficial (avoid overhead for small copies)
    constexpr size_t min_simd_threshold = 32;
    
    if (sample_count < min_simd_threshold)
    {
        // Scalar path for small copies
        for (size_t i = 0; i < sample_count; ++i)
        {
            destination[i] = source[i] & mask;
        }
        return;
    }

#if defined(CHARLS_X86_SIMD_AVAILABLE)
    if constexpr (sizeof(SampleType) == 1)
    {
    #ifdef __AVX2__
        copy_samples_masked_avx2(reinterpret_cast<const uint8_t*>(source),
                                  reinterpret_cast<uint8_t*>(destination),
                                  sample_count, static_cast<uint8_t>(mask));
    #else
        copy_samples_masked_sse2(reinterpret_cast<const uint8_t*>(source),
                                  reinterpret_cast<uint8_t*>(destination),
                                  sample_count, static_cast<uint8_t>(mask));
    #endif
    }
    else if constexpr (sizeof(SampleType) == 2)
    {
    #ifdef __AVX2__
        copy_samples_masked_16bit_avx2(reinterpret_cast<const uint16_t*>(source),
                                        reinterpret_cast<uint16_t*>(destination),
                                        sample_count, static_cast<uint16_t>(mask));
    #else
        copy_samples_masked_16bit_sse2(reinterpret_cast<const uint16_t*>(source),
                                        reinterpret_cast<uint16_t*>(destination),
                                        sample_count, static_cast<uint16_t>(mask));
    #endif
    }
    else
    {
        // Fallback for other sample sizes
        for (size_t i = 0; i < sample_count; ++i)
        {
            destination[i] = source[i] & mask;
        }
    }
#elif defined(CHARLS_ARM_NEON_AVAILABLE)
    if constexpr (sizeof(SampleType) == 1)
    {
        copy_samples_masked_neon(reinterpret_cast<const uint8_t*>(source),
                                  reinterpret_cast<uint8_t*>(destination),
                                  sample_count, static_cast<uint8_t>(mask));
    }
    else if constexpr (sizeof(SampleType) == 2)
    {
        copy_samples_masked_16bit_neon(reinterpret_cast<const uint16_t*>(source),
                                        reinterpret_cast<uint16_t*>(destination),
                                        sample_count, static_cast<uint16_t>(mask));
    }
    else
    {
        // Fallback for other sample sizes
        for (size_t i = 0; i < sample_count; ++i)
        {
            destination[i] = source[i] & mask;
        }
    }
#else
    // Scalar fallback when no SIMD available
    for (size_t i = 0; i < sample_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
#endif
}

} // namespace simd

// =============================================================================
// SIMD Color Transforms
// HP1: R-G, G, B-G (symmetric)
// HP2: R-G, G, B-(R+G)/2
// HP3: G+(B-G+R-G)/4, B-G, R-G
// =============================================================================
namespace simd_color {

#ifdef CHARLS_X86_SIMD_AVAILABLE

/// <summary>
/// HP1 color transform: (R-G+128, G, B-G+128) for 8-bit samples.
/// Processes 16 pixels at a time using SSE2.
/// </summary>

/// <summary>
/// HP1 color transform for 16-bit samples using SSE2.
/// </summary>
inline void transform_hp1_16bit_sse2(const uint16_t* rgb_planar, uint16_t* output,
                                      size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset = _mm_set1_epi16(static_cast<short>(32768));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 8);
    
    for (; i < simd_end; i += 8)
    {
        const __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + stride + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + 2 * stride + i));
        
        const __m128i v1 = _mm_add_epi16(_mm_sub_epi16(r, g), offset);
        const __m128i v3 = _mm_add_epi16(_mm_sub_epi16(b, g), offset);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), v1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * stride + i), v3);
    }
    
    for (; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint16_t>(rgb_planar[i] - rgb_planar[stride + i] + 32768);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint16_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 32768);
    }
}

inline void inverse_hp1_16bit_sse2(const uint16_t* input, uint16_t* rgb_planar,
                                    size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset = _mm_set1_epi16(static_cast<short>(32768));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 8);
    
    for (; i < simd_end; i += 8)
    {
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + stride + i));
        const __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 2 * stride + i));
        
        const __m128i r = _mm_sub_epi16(_mm_add_epi16(v1, g), offset);
        const __m128i b = _mm_sub_epi16(_mm_add_epi16(v3, g), offset);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + i), r);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + 2 * stride + i), b);
    }
    
    for (; i < pixel_count; ++i)
    {
        rgb_planar[i] = static_cast<uint16_t>(input[i] + input[stride + i] - 32768);
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint16_t>(input[2 * stride + i] + input[stride + i] - 32768);
    }
}

inline void transform_hp1_sse2(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset = _mm_set1_epi8(static_cast<char>(128));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        // Load R, G, B planes
        const __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + stride + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + 2 * stride + i));
        
        // v1 = R - G + 128, v2 = G, v3 = B - G + 128
        const __m128i v1 = _mm_add_epi8(_mm_sub_epi8(r, g), offset);
        const __m128i v3 = _mm_add_epi8(_mm_sub_epi8(b, g), offset);
        
        // Store results
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), v1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * stride + i), v3);
    }
    
    // Scalar fallback
    for (; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint8_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 128);
    }
}

/// <summary>
/// Inverse HP1 color transform for 8-bit samples.
/// </summary>
inline void inverse_hp1_sse2(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset = _mm_set1_epi8(static_cast<char>(128));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + stride + i));
        const __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 2 * stride + i));
        
        // R = v1 + G - 128, B = v3 + G - 128
        const __m128i r = _mm_sub_epi8(_mm_add_epi8(v1, g), offset);
        const __m128i b = _mm_sub_epi8(_mm_add_epi8(v3, g), offset);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + i), r);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + 2 * stride + i), b);
    }
    
    for (; i < pixel_count; ++i)
    {
        rgb_planar[i] = static_cast<uint8_t>(input[i] + input[stride + i] - 128);
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint8_t>(input[2 * stride + i] + input[stride + i] - 128);
    }
}

#ifdef __AVX2__
/// <summary>
/// HP1 color transform using AVX2 (32 pixels at a time).
/// </summary>
inline void transform_hp1_avx2(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
    const __m256i offset = _mm256_set1_epi8(static_cast<char>(128));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 32);
    
    for (; i < simd_end; i += 32)
    {
        const __m256i r = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rgb_planar + i));
        const __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rgb_planar + stride + i));
        const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(rgb_planar + 2 * stride + i));
        
        const __m256i v1 = _mm256_add_epi8(_mm256_sub_epi8(r, g), offset);
        const __m256i v3 = _mm256_add_epi8(_mm256_sub_epi8(b, g), offset);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + i), v1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + stride + i), g);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(output + 2 * stride + i), v3);
    }
    
    // Handle remainder with SSE2
    transform_hp1_sse2(rgb_planar + i, output + i, pixel_count - i, stride);
}

/// <summary>
/// Inverse HP1 using AVX2.
/// </summary>
inline void inverse_hp1_avx2(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
    const __m256i offset = _mm256_set1_epi8(static_cast<char>(128));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 32);
    
    for (; i < simd_end; i += 32)
    {
        const __m256i v1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + i));
        const __m256i g = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + stride + i));
        const __m256i v3 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(input + 2 * stride + i));
        
        const __m256i r = _mm256_sub_epi8(_mm256_add_epi8(v1, g), offset);
        const __m256i b = _mm256_sub_epi8(_mm256_add_epi8(v3, g), offset);
        
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(rgb_planar + i), r);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(rgb_planar + stride + i), g);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(rgb_planar + 2 * stride + i), b);
    }
    
    inverse_hp1_sse2(input + i, rgb_planar + i, pixel_count - i, stride);
}
#endif // __AVX2__

#endif // CHARLS_X86_SIMD_AVAILABLE

#ifdef CHARLS_ARM_NEON_AVAILABLE

inline void transform_hp1_neon(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
    const uint8x16_t offset = vdupq_n_u8(128);
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const uint8x16_t r = vld1q_u8(rgb_planar + i);
        const uint8x16_t g = vld1q_u8(rgb_planar + stride + i);
        const uint8x16_t b = vld1q_u8(rgb_planar + 2 * stride + i);
        
        const uint8x16_t v1 = vaddq_u8(vsubq_u8(r, g), offset);
        const uint8x16_t v3 = vaddq_u8(vsubq_u8(b, g), offset);
        
        vst1q_u8(output + i, v1);
        vst1q_u8(output + stride + i, g);
        vst1q_u8(output + 2 * stride + i, v3);
    }
    
    for (; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint8_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 128);
    }
}

inline void inverse_hp1_neon(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
    const uint8x16_t offset = vdupq_n_u8(128);
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const uint8x16_t v1 = vld1q_u8(input + i);
        const uint8x16_t g = vld1q_u8(input + stride + i);
        const uint8x16_t v3 = vld1q_u8(input + 2 * stride + i);
        
        const uint8x16_t r = vsubq_u8(vaddq_u8(v1, g), offset);
        const uint8x16_t b = vsubq_u8(vaddq_u8(v3, g), offset);
        
        vst1q_u8(rgb_planar + i, r);
        vst1q_u8(rgb_planar + stride + i, g);
        vst1q_u8(rgb_planar + 2 * stride + i, b);
    }
    
    for (; i < pixel_count; ++i)
    {
        rgb_planar[i] = input[i] + input[stride + i] - 128;
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = input[2 * stride + i] + input[stride + i] - 128;
    }
}

#endif // CHARLS_ARM_NEON_AVAILABLE

/// <summary>
/// Dispatcher for HP1 transform.
/// </summary>
inline void transform_hp1_simd(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    #ifdef __AVX2__
        transform_hp1_avx2(rgb_planar, output, pixel_count, stride);
    #else
        transform_hp1_sse2(rgb_planar, output, pixel_count, stride);
    #endif
#elif defined(CHARLS_ARM_NEON_AVAILABLE)
    transform_hp1_neon(rgb_planar, output, pixel_count, stride);
#else
    // Scalar fallback
    for (size_t i = 0; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint8_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 128);
    }
#endif
}

inline void transform_hp1_simd(const uint16_t* rgb_planar, uint16_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    transform_hp1_16bit_sse2(rgb_planar, output, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint16_t>(rgb_planar[i] - rgb_planar[stride + i] + 32768);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint16_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 32768);
    }
#endif
}

inline void inverse_hp1_simd(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    #ifdef __AVX2__
        inverse_hp1_avx2(input, rgb_planar, pixel_count, stride);
    #else
        inverse_hp1_sse2(input, rgb_planar, pixel_count, stride);
    #endif
#elif defined(CHARLS_ARM_NEON_AVAILABLE)
    inverse_hp1_neon(input, rgb_planar, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        rgb_planar[i] = static_cast<uint8_t>(input[i] + input[stride + i] - 128);
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint8_t>(input[2 * stride + i] + input[stride + i] - 128);
    }
#endif
}

inline void inverse_hp1_simd(const uint16_t* input, uint16_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    inverse_hp1_16bit_sse2(input, rgb_planar, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        rgb_planar[i] = static_cast<uint16_t>(input[i] + input[stride + i] - 32768);
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint16_t>(input[2 * stride + i] + input[stride + i] - 32768);
    }
#endif
}


#ifdef CHARLS_X86_SIMD_AVAILABLE

/// <summary>
/// HP2 color transform: R-G, G, B-(R+G)/2
/// </summary>
inline void transform_hp2_sse2(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset_128 = _mm_set1_epi8(static_cast<char>(128));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + stride + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + 2 * stride + i));
        
        // v1 = R - G + 128
        const __m128i v1 = _mm_add_epi8(_mm_sub_epi8(r, g), offset_128);
        
        // v3 = B - ((R + G) >> 1) - 128
        // Average: (a+b)>>1 = (a&b) + ((a^b)>>1). 
        // For bytes: (a&b) + (((a^b) >> 1) & 0x7F)
        const __m128i avg_val = _mm_add_epi8(_mm_and_si128(r, g), 
            _mm_and_si128(_mm_srli_epi16(_mm_xor_si128(r, g), 1), _mm_set1_epi8(0x7F)));
         
        const __m128i v3 = _mm_sub_epi8(_mm_sub_epi8(b, avg_val), offset_128);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), v1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * stride + i), v3);
    }
    
    for (; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint8_t>(rgb_planar[2 * stride + i] - ((rgb_planar[i] + rgb_planar[stride + i]) >> 1) - 128);
    }
}

inline void inverse_hp2_sse2(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset_128 = _mm_set1_epi8(static_cast<char>(128));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + stride + i));
        const __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 2 * stride + i));
        
        // R = v1 + G - 128
        const __m128i r = _mm_sub_epi8(_mm_add_epi8(v1, g), offset_128);
        
        // B = v3 + ((R+G)>>1) - 128
        const __m128i avg_val = _mm_add_epi8(_mm_and_si128(r, g), 
            _mm_and_si128(_mm_srli_epi16(_mm_xor_si128(r, g), 1), _mm_set1_epi8(0x7F)));
        
        const __m128i b = _mm_sub_epi8(_mm_add_epi8(v3, avg_val), offset_128);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + i), r);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + 2 * stride + i), b);
    }
    
    for (; i < pixel_count; ++i)
    {
        const uint8_t r = static_cast<uint8_t>(input[i] + input[stride + i] - 128);
        rgb_planar[i] = r;
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint8_t>(input[2 * stride + i] + ((r + input[stride + i]) >> 1) - 128);
    }
}



inline void transform_hp2_16bit_sse2(const uint16_t* rgb_planar, uint16_t* output,
                                      size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset_32768 = _mm_set1_epi16(static_cast<short>(32768));
    const __m128i mask_7FFF = _mm_set1_epi16(0x7FFF);
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 8);
    
    for (; i < simd_end; i += 8)
    {
        const __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + stride + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + 2 * stride + i));
        
        const __m128i v1 = _mm_add_epi16(_mm_sub_epi16(r, g), offset_32768);
        
        // Average (R+G)>>1 without overflow: (R&G) + ((R^G)>>1)
        const __m128i avg = _mm_add_epi16(_mm_and_si128(r, g), 
            _mm_and_si128(_mm_srli_epi16(_mm_xor_si128(r, g), 1), mask_7FFF));
            
        const __m128i v3 = _mm_sub_epi16(_mm_sub_epi16(b, avg), offset_32768);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), v1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * stride + i), v3);
    }
    
    for (; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint16_t>(rgb_planar[i] - rgb_planar[stride + i] + 32768);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint16_t>(rgb_planar[2 * stride + i] - ((rgb_planar[i] + rgb_planar[stride + i]) >> 1) - 32768);
    }
}

inline void inverse_hp2_16bit_sse2(const uint16_t* input, uint16_t* rgb_planar,
                                    size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset_32768 = _mm_set1_epi16(static_cast<short>(32768));
    const __m128i mask_7FFF = _mm_set1_epi16(0x7FFF);
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 8);
    
    for (; i < simd_end; i += 8)
    {
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + stride + i));
        const __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 2 * stride + i));
        
        const __m128i r = _mm_sub_epi16(_mm_add_epi16(v1, g), offset_32768);
        
        const __m128i avg = _mm_add_epi16(_mm_and_si128(r, g), 
            _mm_and_si128(_mm_srli_epi16(_mm_xor_si128(r, g), 1), mask_7FFF));
            
        const __m128i b = _mm_sub_epi16(_mm_add_epi16(v3, avg), offset_32768);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + i), r);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + 2 * stride + i), b);
    }
    
    for (; i < pixel_count; ++i)
    {
        const uint16_t r = static_cast<uint16_t>(input[i] + input[stride + i] - 32768);
        rgb_planar[i] = r;
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint16_t>(input[2 * stride + i] + ((r + input[stride + i]) >> 1) - 32768);
    }
}

/// <summary>
/// HP1 color transform for 16-bit samples using SSE2.
/// </summary>


/// <summary>
/// HP3 color transform
/// </summary>
inline void transform_hp3_sse2(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset_128 = _mm_set1_epi8(static_cast<char>(128));
    const __m128i offset_64 = _mm_set1_epi8(static_cast<char>(64));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const __m128i r = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + i));
        const __m128i g = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + stride + i));
        const __m128i b = _mm_loadu_si128(reinterpret_cast<const __m128i*>(rgb_planar + 2 * stride + i));
        
        // v2 = B - G + 128
        const __m128i v2 = _mm_add_epi8(_mm_sub_epi8(b, g), offset_128);
        // v3 = R - G + 128
        const __m128i v3 = _mm_add_epi8(_mm_sub_epi8(r, g), offset_128);
        
        // v1 = G + ((v2 + v3) >> 2) - 64
        // Use 16-bit expansion for safety
        const __m128i v2_lo = _mm_unpacklo_epi8(v2, _mm_setzero_si128());
        const __m128i v2_hi = _mm_unpackhi_epi8(v2, _mm_setzero_si128());
        const __m128i v3_lo = _mm_unpacklo_epi8(v3, _mm_setzero_si128());
        const __m128i v3_hi = _mm_unpackhi_epi8(v3, _mm_setzero_si128());
        
        __m128i sum_lo = _mm_add_epi16(v2_lo, v3_lo);
        __m128i sum_hi = _mm_add_epi16(v2_hi, v3_hi);
        
        sum_lo = _mm_srai_epi16(sum_lo, 2);
        sum_hi = _mm_srai_epi16(sum_hi, 2);
        
        const __m128i term = _mm_packus_epi16(sum_lo, sum_hi);
        const __m128i v1 = _mm_sub_epi8(_mm_add_epi8(g, term), offset_64);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + i), v1);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + stride + i), v2);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(output + 2 * stride + i), v3);
    }
    
    for (; i < pixel_count; ++i)
    {
        const uint8_t v2 = static_cast<uint8_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 128);
        const uint8_t v3 = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[i] = static_cast<uint8_t>(rgb_planar[stride + i] + ((v2 + v3) >> 2) - 64);
        output[stride + i] = v2;
        output[2 * stride + i] = v3;
    }
}

inline void inverse_hp3_sse2(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
    const __m128i offset_128 = _mm_set1_epi8(static_cast<char>(128));
    const __m128i offset_64 = _mm_set1_epi8(static_cast<char>(64));
    
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const __m128i v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + i));
        const __m128i v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + stride + i));
        const __m128i v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(input + 2 * stride + i));
        
        // g = v1 - ((v3 + v2) >> 2) + 64
        const __m128i v2_lo = _mm_unpacklo_epi8(v2, _mm_setzero_si128());
        const __m128i v2_hi = _mm_unpackhi_epi8(v2, _mm_setzero_si128());
        const __m128i v3_lo = _mm_unpacklo_epi8(v3, _mm_setzero_si128());
        const __m128i v3_hi = _mm_unpackhi_epi8(v3, _mm_setzero_si128());
        
        __m128i sum_lo = _mm_add_epi16(v2_lo, v3_lo);
        __m128i sum_hi = _mm_add_epi16(v2_hi, v3_hi);
        
        sum_lo = _mm_srai_epi16(sum_lo, 2);
        sum_hi = _mm_srai_epi16(sum_hi, 2);
        
        const __m128i term = _mm_packus_epi16(sum_lo, sum_hi);
        const __m128i g = _mm_add_epi8(_mm_sub_epi8(v1, term), offset_64);
        
        // b = v2 + g - 128
        const __m128i b = _mm_sub_epi8(_mm_add_epi8(v2, g), offset_128);
        // r = v3 + g - 128
        const __m128i r = _mm_sub_epi8(_mm_add_epi8(v3, g), offset_128);
        
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + i), r);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + stride + i), g);
        _mm_storeu_si128(reinterpret_cast<__m128i*>(rgb_planar + 2 * stride + i), b);
    }
    
    for (; i < pixel_count; ++i)
    {
        const uint8_t v1 = input[i];
        const uint8_t v2 = input[stride + i];
        const uint8_t v3 = input[2 * stride + i];
        
        const uint8_t g = static_cast<uint8_t>(v1 - ((v3 + v2) >> 2) + 64);
        rgb_planar[stride + i] = g;
        rgb_planar[i] = static_cast<uint8_t>(v3 + g - 128);
        rgb_planar[2 * stride + i] = static_cast<uint8_t>(v2 + g - 128);
    }
}
#endif



inline void transform_hp2_simd(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    transform_hp2_sse2(rgb_planar, output, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint8_t>(rgb_planar[2 * stride + i] - ((rgb_planar[i] + rgb_planar[stride + i]) >> 1) - 128);
    }
#endif
}

inline void inverse_hp2_simd(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    inverse_hp2_sse2(input, rgb_planar, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        const uint8_t r = input[i] + input[stride + i] - 128;
        rgb_planar[i] = r;
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = input[2 * stride + i] + ((r + input[stride + i]) >> 1) - 128;
    }
#endif
}

inline void transform_hp2_simd(const uint16_t* rgb_planar, uint16_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    transform_hp2_16bit_sse2(rgb_planar, output, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        output[i] = static_cast<uint16_t>(rgb_planar[i] - rgb_planar[stride + i] + 32768);
        output[stride + i] = rgb_planar[stride + i];
        output[2 * stride + i] = static_cast<uint16_t>(rgb_planar[2 * stride + i] - ((rgb_planar[i] + rgb_planar[stride + i]) >> 1) - 32768);
    }
#endif
}

inline void inverse_hp2_simd(const uint16_t* input, uint16_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    inverse_hp2_16bit_sse2(input, rgb_planar, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        const uint16_t r = static_cast<uint16_t>(input[i] + input[stride + i] - 32768);
        rgb_planar[i] = r;
        rgb_planar[stride + i] = input[stride + i];
        rgb_planar[2 * stride + i] = static_cast<uint16_t>(input[2 * stride + i] + ((r + input[stride + i]) >> 1) - 32768);
    }
#endif
}

inline void transform_hp3_simd(const uint8_t* rgb_planar, uint8_t* output,
                                size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    transform_hp3_sse2(rgb_planar, output, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        const uint8_t v2 = static_cast<uint8_t>(rgb_planar[2 * stride + i] - rgb_planar[stride + i] + 128);
        const uint8_t v3 = static_cast<uint8_t>(rgb_planar[i] - rgb_planar[stride + i] + 128);
        output[i] = static_cast<uint8_t>(rgb_planar[stride + i] + ((v2 + v3) >> 2) - 64);
        output[stride + i] = v2;
        output[2 * stride + i] = v3;
    }
#endif
}

inline void inverse_hp3_simd(const uint8_t* input, uint8_t* rgb_planar,
                              size_t pixel_count, size_t stride) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    inverse_hp3_sse2(input, rgb_planar, pixel_count, stride);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        const uint8_t v1 = input[i];
        const uint8_t v2 = input[stride + i];
        const uint8_t v3 = input[2 * stride + i];
        
        const uint8_t g = static_cast<uint8_t>(v1 - ((v3 + v2) >> 2) + 64);
        rgb_planar[stride + i] = g;
        rgb_planar[i] = static_cast<uint8_t>(v3 + g - 128);
        rgb_planar[2 * stride + i] = static_cast<uint8_t>(v2 + g - 128);
    }
#endif
}

} // namespace simd_color

// =============================================================================
// SIMD Component Interleaving
// Convert between planar (RRR...GGG...BBB) and interleaved (RGBRGBRGB) formats
// =============================================================================
namespace simd_interleave {

#ifdef CHARLS_X86_SIMD_AVAILABLE

/// <summary>
/// Interleave 3 planes into RGB triplets using SSE2.
/// Processes 16 pixels (48 bytes) at a time.
/// </summary>
inline void interleave_rgb_sse2(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                                 uint8_t* rgb, size_t pixel_count) noexcept
{
    size_t i = 0;
    // SSE2 can process 16 pixels at a time -> 48 bytes output
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        // SSE2 3-channel interleave is complex (no native 3-way shuffle).
        // Using scalar for correctness and clarity. NEON has native vst3.
        for (size_t j = 0; j < 16; ++j)
        {
            rgb[(i + j) * 3 + 0] = r[i + j];
            rgb[(i + j) * 3 + 1] = g[i + j];
            rgb[(i + j) * 3 + 2] = b[i + j];
        }
    }
    
    // Scalar remainder
    for (; i < pixel_count; ++i)
    {
        rgb[i * 3 + 0] = r[i];
        rgb[i * 3 + 1] = g[i];
        rgb[i * 3 + 2] = b[i];
        }
    }


/// <summary>
/// Interleave 3 planes into 16-bit RGB triplets using SSE2.
/// </summary>
inline void interleave_rgb_16bit_sse2(const uint16_t* r, const uint16_t* g, const uint16_t* b,
                                       uint16_t* rgb, size_t pixel_count) noexcept
{
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 8);
    
    for (; i < simd_end; i += 8)
    {
        // Use scalar loop for 16-bit 3-channel interleave
        // (Vector shuffle for 3-channel 16-bit is complex; scalar is clearer and portable)
        for (size_t j = 0; j < 8; ++j)
        {
            rgb[(i + j) * 3 + 0] = r[i + j];
            rgb[(i + j) * 3 + 1] = g[i + j];
            rgb[(i + j) * 3 + 2] = b[i + j];
        }
    }
    
    for (; i < pixel_count; ++i)
    {
        rgb[i * 3 + 0] = r[i];
        rgb[i * 3 + 1] = g[i];
        rgb[i * 3 + 2] = b[i];
    }
}

/// <summary>
/// Deinterleave 16-bit RGB triplets into 3 planes.
/// </summary>
inline void deinterleave_rgb_16bit_sse2(const uint16_t* rgb, uint16_t* r, uint16_t* g, uint16_t* b,
                                         size_t pixel_count) noexcept
{
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 8);
    
    for (; i < simd_end; i += 8)
    {
        // Scalar fallback for complexity
        for (size_t j = 0; j < 8; ++j)
        {
             r[i + j] = rgb[(i + j) * 3 + 0];
             g[i + j] = rgb[(i + j) * 3 + 1];
             b[i + j] = rgb[(i + j) * 3 + 2];
        }
    }
    
    for (; i < pixel_count; ++i)
    {
        r[i] = rgb[i * 3 + 0];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
}


/// <summary>
/// Deinterleave RGB triplets into 3 planes using SSE2.
/// </summary>
inline void deinterleave_rgb_sse2(const uint8_t* rgb, uint8_t* r, uint8_t* g, uint8_t* b,
                                   size_t pixel_count) noexcept
{
    size_t i = 0;
    
    // Use shuffle mask for deinterleaving (SSE3 would be better, use scalar for SSE2)
    // Simplified vectorized approach
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        // Load 48 bytes (16 RGB triplets)
        for (size_t j = 0; j < 16; ++j)
        {
            r[i + j] = rgb[(i + j) * 3 + 0];
            g[i + j] = rgb[(i + j) * 3 + 1];
            b[i + j] = rgb[(i + j) * 3 + 2];
        }
    }
    
    for (; i < pixel_count; ++i)
    {
        r[i] = rgb[i * 3 + 0];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
}

/// <summary>
/// Copy line-interleaved to sample-interleaved (planar to triplet).
/// This is the main operation used in copy_from_line_buffer.
/// </summary>
inline void line_to_sample_3comp_sse2(const uint8_t* source, uint8_t* dest,
                                       size_t pixel_count, size_t pixel_stride) noexcept
{
    // source[i], source[i + stride], source[i + 2*stride] -> dest[i] as triplet
    size_t i = 0;
    
    for (; i < pixel_count; i += 4)
    {
        const size_t remaining = (pixel_count - i < 4) ? pixel_count - i : 4;
        for (size_t j = 0; j < remaining; ++j)
        {
            // Each destination triplet is 3 bytes
            dest[(i + j) * 3 + 0] = source[i + j];
            dest[(i + j) * 3 + 1] = source[i + j + pixel_stride];
            dest[(i + j) * 3 + 2] = source[i + j + 2 * pixel_stride];
        }
    }
}

#endif // CHARLS_X86_SIMD_AVAILABLE

#ifdef CHARLS_ARM_NEON_AVAILABLE

/// <summary>
/// Interleave 3 planes into RGB triplets using NEON.
/// Uses vst3 for efficient 3-way interleave.
/// </summary>
inline void interleave_rgb_neon(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                                 uint8_t* rgb, size_t pixel_count) noexcept
{
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        const uint8x16_t r_vec = vld1q_u8(r + i);
        const uint8x16_t g_vec = vld1q_u8(g + i);
        const uint8x16_t b_vec = vld1q_u8(b + i);
        
        // NEON has native 3-way interleave!
        const uint8x16x3_t rgb_vec = {r_vec, g_vec, b_vec};
        vst3q_u8(rgb + i * 3, rgb_vec);
    }
    
    for (; i < pixel_count; ++i)
    {
        rgb[i * 3 + 0] = r[i];
        rgb[i * 3 + 1] = g[i];
        rgb[i * 3 + 2] = b[i];
    }
}

/// <summary>
/// Deinterleave RGB triplets into 3 planes using NEON.
/// Uses vld3 for efficient 3-way deinterleave.
/// </summary>
inline void deinterleave_rgb_neon(const uint8_t* rgb, uint8_t* r, uint8_t* g, uint8_t* b,
                                   size_t pixel_count) noexcept
{
    size_t i = 0;
    const size_t simd_end = pixel_count - (pixel_count % 16);
    
    for (; i < simd_end; i += 16)
    {
        // NEON has native 3-way deinterleave!
        const uint8x16x3_t rgb_vec = vld3q_u8(rgb + i * 3);
        vst1q_u8(r + i, rgb_vec.val[0]);
        vst1q_u8(g + i, rgb_vec.val[1]);
        vst1q_u8(b + i, rgb_vec.val[2]);
    }
    
    for (; i < pixel_count; ++i)
    {
        r[i] = rgb[i * 3 + 0];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
}

#endif // CHARLS_ARM_NEON_AVAILABLE

/// <summary>
/// Dispatcher for RGB interleaving.
/// </summary>
inline void interleave_rgb_simd(const uint8_t* r, const uint8_t* g, const uint8_t* b,
                                 uint8_t* rgb, size_t pixel_count) noexcept
{
#if defined(CHARLS_ARM_NEON_AVAILABLE)
    // NEON has superior interleave support with vst3
    interleave_rgb_neon(r, g, b, rgb, pixel_count);
#elif defined(CHARLS_X86_SIMD_AVAILABLE)
    interleave_rgb_sse2(r, g, b, rgb, pixel_count);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        rgb[i * 3 + 0] = r[i];
        rgb[i * 3 + 1] = g[i];
        rgb[i * 3 + 2] = b[i];
    }
#endif
}

inline void deinterleave_rgb_simd(const uint8_t* rgb, uint8_t* r, uint8_t* g, uint8_t* b,
                                   size_t pixel_count) noexcept
{
#if defined(CHARLS_ARM_NEON_AVAILABLE)
    deinterleave_rgb_neon(rgb, r, g, b, pixel_count);
#elif defined(CHARLS_X86_SIMD_AVAILABLE)
    deinterleave_rgb_sse2(rgb, r, g, b, pixel_count);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        r[i] = rgb[i * 3 + 0];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
#endif
}




inline void interleave_rgb_simd(const uint16_t* r, const uint16_t* g, const uint16_t* b,
                                 uint16_t* rgb, size_t pixel_count) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    interleave_rgb_16bit_sse2(r, g, b, rgb, pixel_count);
#else
    for (size_t i = 0; i < pixel_count; ++i)
    {
        rgb[i * 3 + 0] = r[i];
        rgb[i * 3 + 1] = g[i];
        rgb[i * 3 + 2] = b[i];
    }
#endif
}

inline void deinterleave_rgb_simd(const uint16_t* rgb, uint16_t* r, uint16_t* g, uint16_t* b,
                                   size_t pixel_count) noexcept
{
#if defined(CHARLS_X86_SIMD_AVAILABLE)
    deinterleave_rgb_16bit_sse2(rgb, r, g, b, pixel_count);
#else
     for (size_t i = 0; i < pixel_count; ++i)
    {
        r[i] = rgb[i * 3 + 0];
        g[i] = rgb[i * 3 + 1];
        b[i] = rgb[i * 3 + 2];
    }
#endif
}

} // namespace simd_interleave

} // namespace charls

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
