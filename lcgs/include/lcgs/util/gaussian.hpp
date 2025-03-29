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

    R = transpose(R); // keep the same with vanilla paper

    Float3x3_T S;
    S[0][0] = scale.x;
    S[1][1] = scale.y;
    S[2][2] = scale.z;
    // compute covariance
    // $\Sigma=R^TS^TSR$
    Float3x3_T M = R * S;
    return M * transpose(M);
}

// q = (x, y, z, w)
template <typename Float3_T, typename Float4_T, typename Float3x3_T>
void calc_cov_backward(
    Float3x3_T dL_dSigma,                    // input
    Float3_T& dL_dscale, Float4_T& dL_dqvec, // output
    // params
    Float3_T scale, Float4_T qvec
)
{
    Float3x3_T R = R_from_qvec<Float4_T, Float3x3_T>(qvec);
    Float3x3_T S;
    S[0][0]      = scale.x;
    S[1][1]      = scale.y;
    S[2][2]      = scale.z;
    Float3x3_T M = S * R;

    Float3x3_T dL_dM = 2.0f * dL_dSigma * M;

    Float3x3_T Rt     = transpose(R);
    Float3x3_T dL_dMt = transpose(dL_dM);
    // here we need rows dot rows

    dL_dscale[0] = dot(Rt[0], dL_dMt[0]);
    dL_dscale[1] = dot(Rt[1], dL_dMt[1]);
    dL_dscale[2] = dot(Rt[2], dL_dMt[2]);
    dL_dMt[0]    = dL_dMt[0] * scale.x;
    dL_dMt[1]    = dL_dMt[1] * scale.y;
    dL_dMt[2]    = dL_dMt[2] * scale.z;

    // transpose
    // $-2 dM_{01} s_{1} z + 2 dM_{02} s_{1} y + 2 dM_{10} s_{2} z - 2 dM_{12} s_{2} x - 2 dM_{20} s_{3} y + 2 dM_{21} s_{3} x$
    dL_dqvec.w =
        2 * qvec.z * (dL_dMt[1][0] - dL_dMt[0][1]) +
        2 * qvec.y * (dL_dMt[0][2] - dL_dMt[2][0]) +
        2 * qvec.x * (dL_dMt[2][1] - dL_dMt[1][2]);

    // $2 dM_{01} s_{1} y + 2 dM_{02} s_{1} z + 2 dM_{10} s_{2} y - 4 dM_{11} s_{2} x - 2 dM_{12} s_{2} w + 2 dM_{20} s_{3} z + 2 dM_{21} s_{3} w - 4 dM_{22} s_{3} x$
    dL_dqvec.x =
        2 * qvec.y * (dL_dMt[0][1] + dL_dMt[1][0]) +
        2 * qvec.z * (dL_dMt[0][2] + dL_dMt[2][0]) +
        2 * qvec.w * (dL_dMt[2][1] - dL_dMt[1][2]) -
        4 * qvec.x * (dL_dMt[2][2] + dL_dMt[1][1]);

    // $-4 dM_{00} s_{1} y + 2 dM_{01} s_{1} x + 2 dM_{02} s_{1} w + 2 dM_{10} s_{2} x + 2 dM_{12} s_{2} z - 2 dM_{20} s_{3} w + 2 dM_{21} s_{3} z - 4 dM_{22} s_{3} y$
    dL_dqvec.y =
        2 * qvec.x * (dL_dMt[0][1] + dL_dMt[1][0]) +
        2 * qvec.w * (dL_dMt[0][2] - dL_dMt[2][0]) +
        2 * qvec.z * (dL_dMt[2][1] + dL_dMt[1][2]) -
        4 * qvec.y * (dL_dMt[2][2] + dL_dMt[0][0]);
    // $-4 dM_{00} s_{1} z - 2 dM_{01} s_{1} w + 2 dM_{02} s_{1} x + 2 dM_{10} s_{2} w - 4 dM_{11} s_{2} z + 2 dM_{12} s_{2} y + 2 dM_{20} s_{3} x + 2 dM_{21} s_{3} y$
    dL_dqvec.z =
        2 * qvec.w * (dL_dMt[1][0] - dL_dMt[0][1]) +
        2 * qvec.x * (dL_dMt[0][2] + dL_dMt[2][0]) +
        2 * qvec.y * (dL_dMt[2][1] + dL_dMt[1][2]) -
        4 * qvec.z * (dL_dMt[1][1] + dL_dMt[0][0]);
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
    W[0]         = view[0].xyz();
    W[1]         = view[1].xyz();
    W[2]         = view[2].xyz();
    W            = transpose(W);
    Float3x3_T T = J * W;
    return T * cov_3d * transpose(T);
}

template <typename Float4_T, typename Float3x3_T>
inline Float3x3_T calc_J(Float4_T camera_primitive, Float4_T p_view)
{
    auto focal_x   = camera_primitive.x;
    auto focal_y   = camera_primitive.y;
    auto tan_fov_x = camera_primitive.z;
    auto tan_fov_y = camera_primitive.w;
    auto t         = p_view.xyz();
    auto limx      = 1.3f * tan_fov_x;
    auto limy      = 1.3f * tan_fov_y;
    auto txtz      = t.x / t.z;
    auto tytz      = t.y / t.z;
    t.x            = clamp(txtz, -limx, limx) * t.z;
    t.y            = clamp(tytz, -limy, limy) * t.z;
    // consider function p = m(t)
    // $p_x=\frac{f_xt_x}{t_z}$
    // $p_y=\frac{f_yt_y}{t_z}$
    // $p_z=1$
    // Calculate the Jacobian of m(t)
    // J =
    // fx/tz, 0.0,  fx*tx/(tz * tz)
    // 0.0,   fy/tz, fy*ty/(tz * tz)
    // 0.0    0.0,   0.0
    Float3x3_T J;
    J[0][0] = focal_x / t.z;
    J[1][1] = focal_y / t.z;
    J[2][0] = focal_x * t.x / (t.z * t.z);
    J[2][1] = focal_y * t.y / (t.z * t.z);
    return J;
}

template <typename Float3_T, typename Float4_T, typename Float3x3_T, typename Float4x4_T>
void proj_cov3d_to_cov2d_backward(
    Float3_T    dL_d_cov_2d, // input
    Float3x3_T& dL_dcov,     // output
    Float4_T p_view, Float4_T camera_primitive,
    Float4x4_T view_matrix
)
{

    Float3x3_T J = calc_J<Float4_T, Float3x3_T>(camera_primitive, p_view);
    Float3x3_T W = make_float3x3(
        view_matrix[0].xyz(),
        view_matrix[1].xyz(),
        view_matrix[2].xyz()
    );
    Float3x3_T T = J * W;

    auto dL_da = dL_d_cov_2d.x;
    auto dL_db = dL_d_cov_2d.y;
    auto dL_dc = dL_d_cov_2d.z;

    // LC matrix is col-major
    // $dC00 = T_{00}^{2} da + T_{00} T_{10} db + T_{10}^{2} dc$
    dL_dcov[0][0] = (T[0][0] * T[0][0] * dL_da + T[0][0] * T[0][1] * dL_db + T[0][1] * T[0][1] * dL_dc);
    // $dC11 = T_{01}^{2} da + T_{01} T_{11} db + T_{11}^{2} dc$
    dL_dcov[1][1] = (T[1][0] * T[1][0] * dL_da + T[1][0] * T[1][1] * dL_db + T[1][1] * T[1][1] * dL_dc);
    // $dC22 = T_{02}^{2} da + T_{02} T_{12} db + T_{12}^{2} dc$
    dL_dcov[2][2] = (T[2][0] * T[2][0] * dL_da + T[2][0] * T[2][1] * dL_db + T[2][1] * T[2][1] * dL_dc);
    // $dC01 = 2 T_{00} T_{01} da + 2 T_{10} T_{11} dc + db \left(T_{00} T_{11} + T_{01} T_{10}\right)$
    dL_dcov[0][1] = 2 * T[0][0] * T[1][0] * dL_da + (T[0][0] * T[1][1] + T[1][0] * T[0][1]) * dL_db + 2 * T[0][1] * T[1][1] * dL_dc;
    // $dC02 = 2 T_{00} T_{02} da + 2 T_{10} T_{12} dc + db \left(T_{00} T_{12} + T_{02} T_{10}\right)$
    dL_dcov[0][2] = 2 * T[0][0] * T[2][0] * dL_da + (T[0][0] * T[2][1] + T[2][0] * T[0][1]) * dL_db + 2 * T[0][1] * T[2][1] * dL_dc;
    //  $dC12 = 2 T_{01} T_{02} da + 2 T_{11} T_{12} dc + db \left(T_{01} T_{12} + T_{02} T_{11}\right)$
    dL_dcov[1][2] = 2 * T[2][0] * T[1][0] * dL_da + (T[1][0] * T[2][1] + T[2][0] * T[1][1]) * dL_db + 2 * T[1][1] * T[2][1];
    // symmetric
    dL_dcov[1][0] = dL_dcov[0][1];
    dL_dcov[2][0] = dL_dcov[0][2];
    dL_dcov[2][1] = dL_dcov[1][2];
}

} // namespace lcgs
