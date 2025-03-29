#pragma once
/**
 * @file concept.hpp
 * @brief The Core Concept
 * @author sailing-innocent
 * @date 2025-03-06
 */

#include <type_traits>

namespace lcgs
{

template <typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;
template <typename T>
concept NumericT = is_numeric_v<T>;

} // namespace lcgs