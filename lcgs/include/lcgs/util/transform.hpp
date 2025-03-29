#pragma once
/**
 * @file transform.h
 * @brief The Graphic Transformation Utilities
 * @author sailing-innocent
 * @date 2025-01-12
 */
#include <luisa/luisa-compute.h>

namespace lcgs
{

template <typename Float_T>
inline Float_T ndc2pix(Float_T ndc, Float_T resolution)
{
    return ((ndc + 1.0f) * resolution - 1.0f) / 2.0f;
}

template <typename Float_T>
inline Float_T pix2ndc(Float_T pix, Float_T resolution)
{
    return 2.0f * pix / resolution - 1.0f;
}

inline luisa::float4x4 get_view_LH(luisa::float3 eye, luisa::float3 dir, luisa::float3 up)
{
    // front vector
    auto f = normalize(dir);
    // side vector
    auto s = normalize(cross(up, f));
    // up vector
    auto u = cross(f, s);
    // s.x s.y s.z -dot(s, eye)
    // u.x u.y u.z -dot(u, eye)
    // f.x f.y f.z -dot(f, eye)
    // 0.0 0.0 0.0 1.0
    auto col0 = luisa::float4{ s.x, u.x, f.x, 0.0f };
    auto col1 = luisa::float4{ s.y, u.y, f.y, 0.0f };
    auto col2 = luisa::float4{ s.z, u.z, f.z, 0.0f };
    auto col3 = luisa::float4{ -dot(s, eye), -dot(u, eye), -dot(f, eye), 1.0f };
    return make_float4x4(col0, col1, col2, col3);
}

inline luisa::float4x4 get_view_RH(luisa::float3 eye, luisa::float3 dir, luisa::float3 up)
{
    // front vector
    auto f = normalize(dir);
    // side vector
    auto s = normalize(-cross(up, f));
    // up vector
    auto u = cross(s, f);
    // the matrix:
    // s.x u.x -f.x 0.0f
    // s.y u.y -f.y 0.0f
    // s.z u.z -f.z 0.0f
    // -dot(s, eye) -dot(u, eye) dot(f, eye) 1.0f

    // s.x s.y s.z -dot(s, eye)
    // u.x u.y u.z -dot(u, eye)
    // -f.x -f.y -f.z dot(f, eye)
    // 0.0 0.0 0.0 1.0
    auto col0 = luisa::float4{ s.x, u.x, -f.x, 0.0f };
    auto col1 = luisa::float4{ s.y, u.y, -f.y, 0.0f };
    auto col2 = luisa::float4{ s.z, u.z, -f.z, 0.0f };
    auto col3 = luisa::float4{ -dot(s, eye), -dot(u, eye), dot(f, eye), 1.0f };
    return make_float4x4(col0, col1, col2, col3);
}

inline luisa::float4x4 get_projection_RH(float fovy, float aspect, float zNear, float zFar)
{
    auto tanHalfFovy = luisa::tan(fovy / 2.0f);
    auto a00         = 1.0f / (aspect * tanHalfFovy);
    auto a11         = 1.0f / tanHalfFovy;
    auto a22         = -(zFar + zNear) / (zFar - zNear);
    auto a23         = -1.0f;
    auto a32         = -2.0f * zFar * zNear / (zFar - zNear);
    auto col0        = luisa::float4{ a00, 0.0f, 0.0f, 0.0f };
    auto col1        = luisa::float4{ 0.0f, a11, 0.0f, 0.0f };
    auto col2        = luisa::float4{ 0.0f, 0.0f, a22, a32 };
    auto col3        = luisa::float4{ 0.0f, 0.0f, a23, 0.0f };

    return transpose(make_float4x4(col0, col1, col2, col3));
}

template <typename Float4_T>
Float4_T qvec_from_aa(Float4_T axis_angle)
{
    Float4_T qvec;
    auto     angle = axis_angle.w;
    auto     axis  = axis_angle.xyz();
    auto     s     = sin(angle / 2.0f);
    qvec.x         = axis.x * s;
    qvec.y         = axis.y * s;
    qvec.z         = axis.z * s;
    qvec.w         = cos(angle / 2.0f);
    return qvec;
}

template <typename Float3_T>
Float3_T rotate_aa(Float3_T axis, Float3_T p)
{
    // Rodrigues' rotation formula
    auto angle = length(axis);
    axis       = normalize(axis);

    auto c  = cos(angle);
    auto s  = sin(angle);
    auto x  = axis.x;
    auto y  = axis.y;
    auto z  = axis.z;
    auto x2 = x * x;
    auto y2 = y * y;
    auto z2 = z * z;
    auto xy = x * y;
    auto xz = x * z;
    auto yz = y * z;
    auto c1 = 1 - c;
    // prot = p * cos(angle) + (u x p) * sin(angle) + u * (u * p) * (1 - cos(angle))
    Float3_T prot;
    prot.x = p.x * (c1 * x2 + c) + p.y * (c1 * xy - z * s) + p.z * (c1 * xz + y * s);
    prot.y = p.x * (c1 * xy + z * s) + p.y * (c1 * y2 + c) + p.z * (c1 * yz - x * s);
    prot.z = p.x * (c1 * xz - y * s) + p.y * (c1 * yz + x * s) + p.z * (c1 * z2 + c);
    return prot;
}

template <typename Float3_T, typename Float3x3_T>
Float3x3_T R_from_aa(Float3_T axis)
{
    Float3x3_T R;
    auto       angle = length(axis);
    axis             = normalize(axis);

    auto c  = cos(angle);
    auto s  = sin(angle);
    auto c1 = 1.0f - c;
    auto x  = axis.x;
    auto y  = axis.y;
    auto z  = axis.z;
    auto x2 = x * x;
    auto y2 = y * y;
    auto z2 = z * z;
    auto xy = x * y;
    auto xz = x * z;
    auto yz = y * z;

    // col 1
    R[0][0] = c1 * x2 + c;
    R[0][1] = c1 * xy + z * s;
    R[0][2] = c1 * xz - y * s;

    R[1][0] = c1 * xy - z * s;
    R[1][1] = c1 * y2 + c;
    R[1][2] = c1 * yz + x * s;

    R[2][0] = c1 * xz + y * s;
    R[2][1] = c1 * yz - x * s;
    R[2][2] = c1 * z2 + c;

    return R;
}

template <typename Float4_T>
Float4_T qvec_mul(Float4_T q1, Float4_T q2)
{
    Float4_T q;
    auto     x1 = q1.x;
    auto     y1 = q1.y;
    auto     z1 = q1.z;
    auto     w1 = q1.w;

    auto x2 = q2.x;
    auto y2 = q2.y;
    auto z2 = q2.z;
    auto w2 = q2.w;

    q.x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2;
    q.y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2;
    q.z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2;
    q.w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2;
    return q;
}

// R
// 1-2z^2-2y^2 & 2xy-2wz & 2xz+2wy
// 2xy+2wz & 1-2x^2-2z^2 & 2yz-2wx
// 2xz-2wy & 2yz+2wx & 1-2x^2-2y^2
// LC is col-major
template <typename Float4_T, typename Float3x3_T>
inline Float3x3_T R_from_qvec(Float4_T q, bool col_major = true)
{
    Float3x3_T R;
    // q = (x, y, z, w)
    auto x = q.x;
    auto y = q.y;
    auto z = q.z;
    auto w = q.w;
    // col 1
    R[0][0] = 1 - 2 * y * y - 2 * z * z;
    R[0][1] = 2 * x * y + 2 * z * w;
    R[0][2] = 2 * x * z - 2 * y * w;
    // col 2
    R[1][0] = 2 * x * y - 2 * z * w;
    R[1][1] = 1 - 2 * x * x - 2 * z * z;
    R[1][2] = 2 * y * z + 2 * x * w;
    // col 3
    R[2][0] = 2 * x * z + 2 * y * w;
    R[2][1] = 2 * y * z - 2 * x * w;
    R[2][2] = 1 - 2 * x * x - 2 * y * y;

    return R;
}

} // namespace lcgs