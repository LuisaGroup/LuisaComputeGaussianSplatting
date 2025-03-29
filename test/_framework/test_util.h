/**
 * @file se_test_util.h
 * @brief Header for sail engine test utility functions and macros
 * @author sailing-innocent
 * @date 2023-09-15
 */

#pragma once

#include <doctest/doctest.h>
#include <span>
#include <concepts>

namespace lcgs::test {

[[nodiscard]] int argc() noexcept;
[[nodiscard]] const char* const* argv() noexcept;
[[nodiscard]] bool float_span_equal(std::span<float> a, std::span<float> b);

// concept for normal value types
template<typename T>
concept CommonValueType = std::is_arithmetic_v<T>;

}// namespace sail::test
