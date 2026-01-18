// Copyright (c) Team CharLS.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include "jpegls_algorithm.hpp"
#include "util.hpp"
#include "assert.hpp"

#include <cstdint>

namespace charls {

/// <summary>
/// JPEG-LS uses arrays of variables: A[0..366], B[0..364], C[0..364] and N[0..366]
/// to maintain the statistic information for the context modeling.
/// As the operations on these variables use the same index it is more efficient to combine A,B,C and N.
/// </summary>
class regular_mode_context final
{
public:
    regular_mode_context() = default;

    explicit regular_mode_context(const int32_t range) noexcept : a_{initialization_value_for_a(range)}
    {
    }

    /// <summary>counter with prediction correction value</summary>
    [[nodiscard]]
    int32_t c() const noexcept
    {
        return c_;
    }

    [[nodiscard]]
    FORCE_INLINE int32_t get_error_correction(const int32_t k) const noexcept
    {
        if (k != 0)
            return 0;

        return bit_wise_sign(2 * b_ + n_ - 1);
    }

    /// <summary>Code segment A.12 – Variables update. ISO 14495-1, page 22</summary>
    FORCE_INLINE void update_variables_and_bias(const int32_t error_value, const int32_t near_lossless,
                                                const int32_t reset_threshold)
    {
        ASSERT(n_ != 0);

        a_ += std::abs(error_value);
        b_ += error_value * (2 * near_lossless + 1);

        if (constexpr int limit{65536 * 256}; UNLIKELY(a_ >= limit || std::abs(b_) >= limit))
            impl::throw_jpegls_error(jpegls_errc::invalid_data);

        if (n_ == reset_threshold)
        {
            a_ >>= 1;
            b_ >>= 1;
            n_ >>= 1;
        }

        ++n_;
        ASSERT(n_ != 0);

        // This part is from: Code segment A.13 – Update of bias-related variables B[Q] and C[Q]
        constexpr int32_t max_c{127};  // MAX_C: maximum allowed value of C[0..364]. ISO 14495-1, section 3.3
        constexpr int32_t min_c{-128}; // MIN_C: Minimum allowed value of C[0..364]. ISO 14495-1, section 3.3
        if (b_ + n_ <= 0)
        {
            b_ += n_;
            if (b_ <= -n_)
            {
                b_ = -n_ + 1;
            }
            if (c_ > min_c)
            {
                --c_;
            }
        }
        else if (b_ > 0)
        {
            b_ -= n_;
            if (b_ > 0)
            {
                b_ = 0;
            }
            if (c_ < max_c)
            {
                ++c_;
            }
        }
    }

    /// <summary>
    /// Optimized version of update_variables_and_bias for the case where error_value == 0.
    /// This is common in MRI / Medical imaging (smooth 16-bit blocks).
    /// </summary>
    FORCE_INLINE void update_variables_and_bias_zero_error(const int32_t reset_threshold)
    {
        ASSERT(n_ != 0);

        // a_ += 0; -> Optimized away
        // b_ += 0; -> Optimized away

        if (n_ == reset_threshold)
        {
            a_ >>= 1;
            b_ >>= 1;
            n_ >>= 1;
        }

        ++n_;

        // Optimized bias update for b_ unchanged case
        constexpr int32_t max_c{127};
        constexpr int32_t min_c{-128};

        if (b_ + n_ <= 0)
        {
            b_ += n_;
            if (b_ <= -n_)
            {
                b_ = -n_ + 1;
            }
            if (c_ > min_c)
            {
                --c_;
            }
        }
        else if (b_ > 0)
        {
            b_ -= n_;
            if (b_ > 0)
            {
                b_ = 0;
            }
            if (c_ < max_c)
            {
                ++c_;
            }
        }
        // else: b_ <= 0 && b_ + n_ > 0 -> stable state, no updates needed
    }

    /// <summary>
    /// Computes the Golomb coding parameter using the algorithm as defined in ISO 14495-1, code segment A.10
    /// Optimized with CLZ (count leading zeros) for O(1) computation instead of O(k) loop.
    /// </summary>
    [[nodiscard]]
    FORCE_INLINE int32_t compute_golomb_coding_parameter() const
    {
        int32_t k{};
        
        // Early exit for common case
        if (n_ << k >= a_)
            return k;

        // CLZ-based estimation: k ≈ floor(log2(a_)) - floor(log2(n_))
        const int k_est{charls::countl_zero(static_cast<uint32_t>(n_)) -
                        charls::countl_zero(static_cast<uint32_t>(a_))};
        
        k = k_est > 0 ? k_est : 0;
        
        // CLZ gives estimate, may need +1 adjustment
        if ((n_ << k) < a_)
        {
            ++k;
        }

        // Safety verification loop (runs 0 times with correct CLZ)
        for (; n_ << k < a_ && k < max_k_value; ++k)
        {
        }

        if (UNLIKELY(k == max_k_value))
            impl::throw_jpegls_error(jpegls_errc::invalid_data);

        return k;
    }

private:
    // Initialize with the default values as defined in ISO 14495-1, A.8, step 1.d.
    int32_t a_{};
    int32_t b_{};
    int32_t c_{};
    int32_t n_{1};
};

} // namespace charls
