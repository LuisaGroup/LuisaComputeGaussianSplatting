#pragma once

#include <luisa/luisa-compute.h>
#include <luisa/gui/imgui_window.h>
#include <lcgs/util/camera.h>

namespace lcgs
{

class Display
{

private:
    using TransposeShader = luisa::compute::Shader2D<luisa::compute::Buffer<float>, luisa::compute::Image<float>>;
    luisa::compute::Stream&      _stream;
    Camera&                      _camera;
    luisa::float3&               _bg_color;
    luisa::compute::ImGuiWindow  _window;
    luisa::compute::Image<float> _framebuffer;
    uint64_t                     _framebuffer_handle;
    TransposeShader              _transpose_shader;

public:
    Display(luisa::compute::Device& device, luisa::compute::Stream& stream, Camera& camera, luisa::float3& bg_color, luisa::compute::uint2 resolution) noexcept;
    ~Display() noexcept;
    [[nodiscard]] bool is_running() const noexcept;
    void               present(luisa::compute::BufferView<float> d_img) noexcept;
};

} // namespace lcgs
