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

// float3 cov2d [0][0] [0][1] [1][1]
template <typename Float3_T, typename Float_T>
Float3_T cov2d_to_conic(Float3_T cov_2d)
{
    Float_T  det     = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
    Float_T  inv_det = 1.0f / (det + 1e-6f);
    Float3_T conic   = inv_det * Float3_T(cov_2d.z, -cov_2d.y, cov_2d.x);
    return conic;
}

// float3 cov2d [0][0] [0][1] [1][1]
// float3 dL_d_conic [0][0] [0][1] [1][1]
// return float3 dL_d_cov2d [0][0] [0][1] [1][1]
template <typename Float3_T, typename Float_T>
Float3_T cov2d_to_conic_backward(Float3_T cov_2d, Float3_T dL_d_conic)
{
    Float_T  det       = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
    Float_T  det_inv   = 1.0f / (det + 1e-6f);
    Float_T  det_inv_2 = det_inv * det_inv;
    Float3_T dL_d_cov2d;
    dL_d_cov2d.x = det_inv_2 * (-cov_2d.z * cov_2d.z * dL_d_conic.x + 2 * cov_2d.y * cov_2d.z * dL_d_conic.y - cov_2d.y * cov_2d.y * dL_d_conic.z);
    dL_d_cov2d.z = det_inv_2 * (-cov_2d.y * cov_2d.y * dL_d_conic.x + 2 * cov_2d.x * cov_2d.y * dL_d_conic.y - cov_2d.x * cov_2d.x * dL_d_conic.z);
    dL_d_cov2d.y = det_inv_2 * 2 * (cov_2d.y * cov_2d.z * dL_d_conic.x - (cov_2d.x * cov_2d.z + cov_2d.y * cov_2d.y) * dL_d_conic.y + cov_2d.x * cov_2d.y * dL_d_conic.z);
    return dL_d_cov2d;
}

// q = (x, y, z, w)
template <typename Float3_T, typename Float4_T, typename Float3x3_T>
Float3x3_T calc_cov(Float3_T scale, Float4_T qvec)
{
    // $\mathbf{R}=\left[\begin{matrix}1-2x^2-2y^2 & 2xy-2rz & 2xz+2ry \\ 2xy+2rz & 1-2x^2-2z^2 & 2yz-2rx \\ 2xz-2ry & 2yz+2rx & 1-2x^2-2y^2 \end{matrix}\right]$
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

template <typename Float3x3_T, typename Float4x4_T, typename Float3_T>
inline Float3x3_T ewasplat_cov(Float3x3_T cov_3d, Float3_T t, Float4x4_T view)
{
    Float3x3_T J;
    J[0][0] = 1.0f / t.z;
    J[1][1] = 1.0f / t.z;
    J[2][0] = (-t.x) / (t.z * t.z);
    J[2][1] = (-t.y) / (t.z * t.z);
    Float3x3_T W;
    W[0] = view[0].xyz();
    W[1] = view[1].xyz();
    W[2] = view[2].xyz();

    // W = transpose(W);
    Float3x3_T T = J * W;
    return T * cov_3d * transpose(T);
}

} // namespace lcgs
