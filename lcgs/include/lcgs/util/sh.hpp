#pragma once
/**
 * @file sh.hpp
 * @brief Spherical Harmonics Utils
 * @author sailing-innocent
 * @date 2025-01-12
 */

namespace lcgs
{

constexpr float SH_C0 = 0.28209479177387814f; // $\frac{1}{2}\sqrt{\frac{1}{\pi}}$
constexpr float SH_C1 = 0.4886025119029199f;
// $a=\frac{1}{2}\sqrt{\frac{3}{\pi}}$, $C_{10}=\frac{1}{\sqrt{2}}a$,
// $C_{11}=a$, $C_{12}=-\frac{1}{\sqrt{2}}a$
constexpr float SH_C2[5] = { 1.0925484305920792f,
                             -1.0925484305920792f,
                             0.31539156525252005f,
                             -1.0925484305920792f,
                             0.5462742152960396f };

constexpr float SH_C3[7] = { -0.5900435899266435f,
                             2.890611442640554f,
                             -0.4570457994644658f,
                             0.3731763325901154f,
                             -0.4570457994644658f,
                             1.445305721320277f,
                             -0.5900435899266435f };

template <typename Float3_T>
Float3_T compute_color_from_sh_level_0(Float3_T sh_00)
{
    return sh_00 * SH_C0;
}

template <typename Float3_T>
void compute_color_from_sh_level_0_backward(Float3_T dL_dcolor, Float3_T& dL_d_sh00)
{
    dL_d_sh00 = dL_dcolor * SH_C0;
}

template <typename Float3_T>
Float3_T compute_color_from_sh_level_1(Float3_T dir, Float3_T sh_10, Float3_T sh_11, Float3_T sh_12)
{
    auto x = dir.x;
    auto y = dir.y;
    auto z = dir.z;
    // ? is it right?
    return -SH_C1 * (sh_10 * y - sh_11 * z + sh_12 * x);
}

template <typename Float3_T>
void compute_color_from_sh_level_1_backward(Float3_T dL_dcolor, // input
                                            Float3_T dir,       // param
                                            // output
                                            Float3_T& dL_d_sh10, Float3_T& dL_d_sh11, Float3_T& dL_d_sh12, Float3_T& dL_d_dir)
{
    auto x    = dir.x;
    auto y    = dir.y;
    auto z    = dir.z;
    dL_d_sh10 = -SH_C1 * y * dL_dcolor;
    dL_d_sh11 = SH_C1 * z * dL_dcolor;
    dL_d_sh12 = -SH_C1 * x * dL_dcolor;
    // TODO dL_d_dir
}

template <typename Float3_T>
Float3_T compute_color_from_sh_level_2(
    Float3_T dir, Float3_T sh_20, Float3_T sh_21, Float3_T sh_22, Float3_T sh_23, Float3_T sh_24
)
{
    auto x  = dir.x;
    auto y  = dir.y;
    auto z  = dir.z;
    auto xx = x * x;
    auto yy = y * y;
    auto yz = y * z;
    auto zz = z * z;
    auto zx = z * x;
    auto xy = x * y;
    return SH_C2[0] * xy * sh_20 + SH_C2[1] * yz * sh_21 +
           SH_C2[2] * (2.0f * zz - xx - yy) * sh_22 + SH_C2[3] * zx * sh_23 +
           SH_C2[4] * (xx - yy) * sh_24;
}

template <typename Float3_T>
inline void compute_color_from_sh_level_2_backward(
    Float3_T dL_dcolor, // input
    Float3_T dir,       // param
    // output
    Float3_T& dL_d_sh20,
    Float3_T& dL_d_sh21,
    Float3_T& dL_d_sh22,
    Float3_T& dL_d_sh23,
    Float3_T& dL_d_sh24,
    Float3_T& dL_d_dir
)
{

    auto x  = dir.x;
    auto y  = dir.y;
    auto z  = dir.z;
    auto xx = x * x;
    auto yy = y * y;
    auto yz = y * z;
    auto zz = z * z;
    auto zx = z * x;
    auto xy = x * y;

    dL_d_sh20 = SH_C2[0] * xy * dL_dcolor;
    dL_d_sh21 = SH_C2[1] * yz * dL_dcolor;
    dL_d_sh22 = SH_C2[2] * (2.0f * zz - xx - yy) * dL_dcolor;
    dL_d_sh23 = SH_C2[3] * zx * dL_dcolor;
    dL_d_sh24 = SH_C2[4] * (xx - yy) * dL_dcolor;

    // TODO dL_d_dir
}

template <typename Float3_T>
Float3_T compute_color_from_sh_level_3(Float3_T dir, Float3_T sh_30, Float3_T sh_31, Float3_T sh_32, Float3_T sh_33, Float3_T sh_34, Float3_T sh_35, Float3_T sh_36)
{
    auto x  = dir.x;
    auto y  = dir.y;
    auto z  = dir.z;
    auto xx = x * x;
    auto yy = y * y;
    auto yz = y * z;
    auto zz = z * z;
    auto zx = z * x;
    auto xy = x * y;
    return SH_C3[0] * y * (3.0f * xx - yy) * sh_30 +
           SH_C3[1] * xy * z * sh_31 +
           SH_C3[2] * y * (4.0f * zz - xx - yy) * sh_32 +
           SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh_33 +
           SH_C3[4] * x * (4.0f * zz - xx - yy) * sh_34 +
           SH_C3[5] * z * (xx - yy) * sh_35 +
           SH_C3[6] * x * (xx - 3.0f * yy) * sh_36;
}

template <typename Float3_T>
void compute_color_from_sh_level_3_backward(Float3_T dL_dcolor, // input
                                            Float3_T dir,       // param
                                            // output
                                            Float3_T& dL_d_sh30, Float3_T& dL_d_sh31, Float3_T& dL_d_sh32, Float3_T& dL_d_sh33, Float3_T& dL_d_sh34, Float3_T& dL_d_sh35, Float3_T& dL_d_sh36, Float3_T& dL_d_dir)
{
    auto x  = dir.x;
    auto y  = dir.y;
    auto z  = dir.z;
    auto xx = x * x;
    auto yy = y * y;
    auto yz = y * z;
    auto zz = z * z;
    auto zx = z * x;
    auto xy = x * y;

    dL_d_sh30 = SH_C3[0] * y * (3.0f * xx - yy) * dL_dcolor;
    dL_d_sh31 = SH_C3[1] * xy * z * dL_dcolor;
    dL_d_sh32 = SH_C3[2] * y * (4.0f * zz - xx - yy) * dL_dcolor;
    dL_d_sh33 = SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * dL_dcolor;
    dL_d_sh34 = SH_C3[4] * x * (4.0f * zz - xx - yy) * dL_dcolor;
    dL_d_sh35 = SH_C3[5] * z * (xx - yy) * dL_dcolor;
    dL_d_sh36 = SH_C3[6] * x * (xx - 3.0f * yy) * dL_dcolor;

    // TODO dL_d_dir
}

inline void compute_sh_from_color(
    const float color,
    float&      sh
)
{
    sh = (color - 0.5f) / SH_C0;
}

} // namespace lcgs