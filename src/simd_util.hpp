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

} // namespace simd

} // namespace charls
