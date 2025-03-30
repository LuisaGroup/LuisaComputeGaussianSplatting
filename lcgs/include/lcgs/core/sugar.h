#pragma once
/**
 * @file sugar.h
 * @brief The Sugar to simplify code
 * @author sailing-innocent
 * @date 2024-11-21
 */

#include <luisa/dsl/sugar.h>

namespace lcgs
{

inline luisa::compute::Float2 read_float2(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx) noexcept
{
    return make_float2(buffer.read(2 * idx + 0), buffer.read(2 * idx + 1));
}

inline void write_float2(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx, luisa::compute::Float2 value) noexcept
{
    buffer.write(2 * idx + 0, value.x);
    buffer.write(2 * idx + 1, value.y);
}

inline luisa::compute::Float3 read_float3(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx) noexcept
{
    return make_float3(buffer.read(3 * idx + 0), buffer.read(3 * idx + 1), buffer.read(3 * idx + 2));
}

inline void write_float3(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx, luisa::compute::Float3 value) noexcept
{
    buffer.write(3 * idx + 0, value.x);
    buffer.write(3 * idx + 1, value.y);
    buffer.write(3 * idx + 2, value.z);
}

inline void atomic_add_float3(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx, luisa::compute::Float3 value) noexcept
{
    buffer.atomic(idx * 3 + 0).fetch_add(value.x);
    buffer.atomic(idx * 3 + 1).fetch_add(value.y);
    buffer.atomic(idx * 3 + 2).fetch_add(value.z);
}

inline luisa::compute::Float4 read_float4(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx) noexcept
{
    return make_float4(buffer.read(4 * idx + 0), buffer.read(4 * idx + 1), buffer.read(4 * idx + 2), buffer.read(4 * idx + 3));
}

inline void write_float4(luisa::compute::BufferVar<float>& buffer, luisa::compute::Int idx, luisa::compute::Float4 value) noexcept
{
    buffer.write(4 * idx + 0, value.x);
    buffer.write(4 * idx + 1, value.y);
    buffer.write(4 * idx + 2, value.z);
    buffer.write(4 * idx + 3, value.w);
}

inline int block_aligned(int x, int block)
{
    return (x / block + (x % block ? 1 : 0)) * block;
}

template <typename T>
luisa::compute::BufferView<T> soa_element_buf_view(luisa::compute::SOAView<T> view)
{
    return view.buffer()
        .subview(
            view.soa_offset() + view.element_offset() * view.element_stride,
            view.element_size()
        )
        .template as<T>();
}

} // namespace lcgs