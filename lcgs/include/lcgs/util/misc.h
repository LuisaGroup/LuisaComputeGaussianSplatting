/**
 * @file bit_helper.h
 * @brief Bit Helper
 * @author sailing-innocent
 * @date 2025-01-11
 */

#include <cmath>
#include "lcgs/config.h"

namespace lcgs
{

uint32_t     get_higher_msb(uint32_t n);
inline float to_radius(float degree) { return degree * 0.0174532925f; }
inline int   imax(int a, int b) { return a > b ? a : b; }
inline bool  is_power_of_two(int x) { return (x & (x - 1)) == 0; }
inline int   floor_pow_2(int n)
{
#ifdef WIN32
    return 1 << (int)logb((float)n);
#else
    int exp;
    frexp((float)n, &exp);
    return 1 << (exp - 1);
#endif
}

inline float radians(float degree)
{
    return degree * 0.017453292519943295769236907684886f;
}

template <typename Vec3T>
Vec3T cross(Vec3T v1, Vec3T v2)
{
    return Vec3T(
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0]
    );
}

template <typename Vec3T>
Vec3T normalize(Vec3T v)
{
    float length = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    if (length > 0.0f)
    {
        v /= length;
    }
    return v;
}

template <typename Float_T, typename Int_T>
inline Float_T _tab10(Int_T i, Int_T j)
{
    constexpr float tab10_data[30] = {
        0.12, 0.47, 0.71, 1.00, 0.50, 0.05, 0.17, 0.63, 0.17, 0.84, 0.15, 0.16, 0.58, 0.40, 0.74, 0.55, 0.34, 0.29, 0.89, 0.47, 0.76, 0.50, 0.50, 0.50, 0.74, 0.74, 0.13, 0.09, 0.75, 0.81
    };
    return tab10_data[3 * i + j];
}

} // namespace lcgs
