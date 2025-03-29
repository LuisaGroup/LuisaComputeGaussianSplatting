#pragma once
/**
 * @file gaussian.hpp
 * @brief The Gaussian Distribution Utilities
 * @author sailing-innocent
 * @date 2025-01-12
 */

#include "transform.hpp"

namespace lcgs
{

// q = (x, y, z, w)
template <typename Float3_T, typename Float4_T, typename Float3x3_T>
Float3x3_T calc_cov(Float3_T scale, Float4_T qvec)
{
    // LuisaCompute is Col-Major
    Float3x3_T R = R_from_qvec<Float4_T, Float3x3_T>(qvec);

    // R = transpose(R); // keep the same with vanilla paper
    Float3x3_T S;
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;
    // compute covariance
    // $\Sigma=R^TS^TSR$
    Float3x3_T M = R * S;
    return M * transpose(M);
}

// project cov to screen space
template <typename Float3x3_T, typename Float4x4_T, typename Float3_T>
inline Float3x3_T ewasplat_cov(Float3x3_T cov_3d, Float3_T t, Float4x4_T view)
{
    Float3x3_T J;
    J[0][0] = 1.0f / t.z;
    J[1][1] = 1.0f / t.z;
    J[0][2] = (-t.x) / (t.z * t.z);
    J[1][2] = (-t.y) / (t.z * t.z);

    Float3x3_T W;
    W[0] = view[0].xyz();
    W[1] = view[1].xyz();
    W[2] = view[2].xyz();

    W = transpose(W);

    Float3x3_T T = W * J;
    return transpose(T) * cov_3d * T;
}

// project cov to image space directly
template <typename Float3x3_T, typename Float4x4_T, typename Float3_T, typename Float_T>
inline Float3x3_T ewasplat_cov_focal(Float3x3_T cov_3d, Float3_T t, Float4x4_T view, Float_T focalx, Float_T focaly)
{
    Float3x3_T J;
    J[0][0] = focalx / t.z;
    J[1][1] = focaly / t.z;
    J[0][2] = (-focalx * t.x) / (t.z * t.z);
    J[1][2] = (-focaly * t.y) / (t.z * t.z);

    Float3x3_T W;
    W[0] = view[0].xyz();
    W[1] = view[1].xyz();
    W[2] = view[2].xyz();

    W = transpose(W);

    Float3x3_T T = W * J;
    return transpose(T) * cov_3d * T;
}

} // namespace lcgs
