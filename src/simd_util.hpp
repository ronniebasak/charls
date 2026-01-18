// Copyright (c) Team CharLS.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// SIMD utilities for CharLS
// Provides vectorized operations for line buffer processing

#include <cstddef>
#include <cstdint>
#include <cstring>

// =============================================================================
// Platform Detection
// =============================================================================

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define CHARLS_X86_SIMD_AVAILABLE 1
#include <immintrin.h>
#elif defined(__aarch64__) || defined(_M_ARM64)
#define CHARLS_ARM_NEON_AVAILABLE 1
#include <arm_neon.h>
#endif

namespace charls {

// =============================================================================
// Runtime CPU Feature Detection
// =============================================================================
namespace simd {

#ifdef CHARLS_X86_SIMD_AVAILABLE

/// <summary>
/// Check if SSE2 is available at runtime.
/// On x86-64, SSE2 is always available.
/// </summary>
[[nodiscard]]
inline bool cpu_supports_sse2() noexcept
{
#ifdef __SSE2__
    return true;  // Compile-time guarantee on x86-64
#else
    // For 32-bit x86, runtime detection would go here
    return false;
#endif
}

/// <summary>
/// Check if AVX2 is available at runtime.
/// </summary>
[[nodiscard]]
inline bool cpu_supports_avx2() noexcept
{
#ifdef __AVX2__
    return true;  // Compile-time guarantee when -mavx2 is used
#else
    return false;
#endif
}

#endif // CHARLS_X86_SIMD_AVAILABLE

// =============================================================================
// SIMD Masked Copy Operations
// =============================================================================

#ifdef CHARLS_X86_SIMD_AVAILABLE

/// <summary>
/// SIMD-optimized masked copy for 8-bit samples using SSE2.
/// Processes 16 bytes at a time.
/// </summary>
inline void copy_samples_masked_sse2(const uint8_t* source, uint8_t* destination,
                                      size_t byte_count, uint8_t mask) noexcept
{
    const __m128i mask_vec = _mm_set1_epi8(static_cast<char>(mask));
    
    size_t i = 0;
    const size_t simd_end = byte_count - (byte_count % 16);
    for (; i < simd_end; i += 16)
    {
        // Cast through void* to avoid -Wcast-align (loadu is explicitly unaligned)
        const __m128i data = _mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(source + i)));
        const __m128i result = _mm_and_si128(data, mask_vec);
        _mm_storeu_si128(static_cast<__m128i*>(static_cast<void*>(destination + i)), result);
    }
    
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
    const size_t simd_end = sample_count - (sample_count % 8);
    for (; i < simd_end; i += 8)
    {
        // Cast through void* to avoid -Wcast-align (loadu is explicitly unaligned)
        const __m128i data = _mm_loadu_si128(static_cast<const __m128i*>(static_cast<const void*>(source + i)));
        const __m128i result = _mm_and_si128(data, mask_vec);
        _mm_storeu_si128(static_cast<__m128i*>(static_cast<void*>(destination + i)), result);
    }
    
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
    const size_t simd_end = byte_count - (byte_count % 32);
    for (; i < simd_end; i += 32)
    {
        // Cast through void* to avoid -Wcast-align (loadu is explicitly unaligned)
        const __m256i data = _mm256_loadu_si256(static_cast<const __m256i*>(static_cast<const void*>(source + i)));
        const __m256i result = _mm256_and_si256(data, mask_vec);
        _mm256_storeu_si256(static_cast<__m256i*>(static_cast<void*>(destination + i)), result);
    }
    
    for (; i < byte_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
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
    const size_t simd_end = sample_count - (sample_count % 16);
    for (; i < simd_end; i += 16)
    {
        // Cast through void* to avoid -Wcast-align (loadu is explicitly unaligned)
        const __m256i data = _mm256_loadu_si256(static_cast<const __m256i*>(static_cast<const void*>(source + i)));
        const __m256i result = _mm256_and_si256(data, mask_vec);
        _mm256_storeu_si256(static_cast<__m256i*>(static_cast<void*>(destination + i)), result);
    }
    
    for (; i < sample_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
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
    const size_t simd_end = byte_count - (byte_count % 16);
    for (; i < simd_end; i += 16)
    {
        const uint8x16_t data = vld1q_u8(source + i);
        const uint8x16_t result = vandq_u8(data, mask_vec);
        vst1q_u8(destination + i, result);
    }
    
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
    const size_t simd_end = sample_count - (sample_count % 8);
    for (; i < simd_end; i += 8)
    {
        const uint16x8_t data = vld1q_u16(source + i);
        const uint16x8_t result = vandq_u16(data, mask_vec);
        vst1q_u16(destination + i, result);
    }
    
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
#if defined(CHARLS_ARM_NEON_AVAILABLE)
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
#elif defined(CHARLS_X86_SIMD_AVAILABLE)
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
#else
    // Scalar fallback
    for (size_t i = 0; i < sample_count; ++i)
    {
        destination[i] = source[i] & mask;
    }
#endif
}

} // namespace simd

} // namespace charls
