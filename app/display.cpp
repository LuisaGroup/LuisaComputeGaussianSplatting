//
// Created by mike on 3/31/25.
//

#include <imgui.h>
#include "display.h"

namespace lcgs
{

using namespace luisa;
using namespace luisa::compute;

Display::Display(luisa::compute::Device& device, luisa::compute::Stream& stream, Camera& camera, luisa::float3& bg_color, luisa::compute::uint2 resolution) noexcept
    : _stream{ stream }
    , _camera{ camera }
    , _bg_color{ bg_color }
    , _framebuffer_handle{}
{
    ImGuiWindow::Config config{ .size = resolution, .resizable = false, .vsync = true, .hdr = false, .ssaa = false, .docking = true, .multi_viewport = false, .back_buffers = 2 };
    _window             = ImGuiWindow{ device, stream, "Display", config };
    _framebuffer        = device.create_image<float>(PixelStorage::BYTE4, resolution, 1);
    _framebuffer_handle = _window.register_texture(_framebuffer, Sampler::linear_point_edge());
    _transpose_shader   = device.compile<2>([](BufferFloat d_img, ImageFloat fb) noexcept {
        auto size = dispatch_size().xy();
        auto n    = size.x * size.y;
        auto p    = dispatch_id().xy();
        // convert from CxHxW to WxHxC
        auto c0 = d_img.read(n * 0u + p.y * size.x + p.x);
        auto c1 = d_img.read(n * 1u + p.y * size.x + p.x);
        auto c2 = d_img.read(n * 2u + p.y * size.x + p.x);
        fb.write(p, make_float4(c0, c1, c2, 1.f));
    });
}

Display::~Display() noexcept = default;

bool Display::is_running() const noexcept
{
    return !_window.should_close();
}

void Display::present(luisa::compute::BufferView<float> d_img) noexcept
{
    _stream << _transpose_shader(d_img, _framebuffer).dispatch(_framebuffer.size());
    _window.with_frame([&] {
        auto viewport = ImGui::GetMainViewport();
        auto p_min    = ImVec2{ viewport->Pos.x, viewport->Pos.y };
        auto p_max    = p_min +
                     ImVec2{ static_cast<float>(_framebuffer.size().x),
                             static_cast<float>(_framebuffer.size().y) };
        ImGui::GetBackgroundDrawList()->AddImage(_framebuffer_handle, p_min, p_max, { 0.f, 1.f }, { 1.f, 0.f });

        ImGui::Begin("Control");
        {
            ImGui::ColorPicker3("Background", &_bg_color.x, ImGuiColorEditFlags_Float);
        }
        ImGui::End();
    });
}

} // namespace lcgs