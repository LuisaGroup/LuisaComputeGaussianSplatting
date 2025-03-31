//
// Created by mike on 3/31/25.
//

#ifndef IMGUI_DEFINE_MATH_OPERATORS
    #define IMGUI_DEFINE_MATH_OPERATORS
#endif

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
    , _camera_move_speed{ 1.f }
    , _camera_rotate_speed{ 2.f }
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
        auto size = ImVec2{ static_cast<float>(_framebuffer.size().x),
                            static_cast<float>(_framebuffer.size().y) };

        auto camera_dirty = false;
        if (!ImGui::GetIO().WantCaptureMouse)
        {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                _camera.front += .5f * _camera_rotate_speed * (delta.x / size.x) * _camera.right;
                _camera.front -= .5f * _camera_rotate_speed * (delta.y / size.y) * _camera.up;
                camera_dirty = true;
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
            {
                auto p_now  = ImGui::GetMousePos() - ImVec2{ .5f, .5f } * size;
                auto p_prev = p_now - ImGui::GetMouseDragDelta(ImGuiMouseButton_Right);
                auto v_now  = luisa::normalize(luisa::make_float3(p_now.x, p_now.y, 0.f));
                auto v_prev = luisa::normalize(luisa::make_float3(p_prev.x, p_prev.y, 0.f));
                auto c      = luisa::cross(v_now, v_prev);
                auto angle  = luisa::sign(luisa::dot(c, luisa::make_float3(0.f, 0.f, 1.f))) *
                             luisa::acos(luisa::dot(v_now, v_prev));
                if (!luisa::isnan(angle))
                {
                    auto m       = luisa::make_float3x3(luisa::rotation(_camera.front, angle));
                    _camera.up   = m * _camera.up;
                    camera_dirty = true;
                    ImGui::ResetMouseDragDelta(ImGuiMouseButton_Right);
                }
            }
        }

        if (ImGui::IsKeyPressed(ImGuiKey_Escape))
        {
            _window.set_should_close(true);
        }

        if (!ImGui::GetIO().WantCaptureKeyboard)
        {

            auto dt = ImGui::GetIO().DeltaTime;
            if (ImGui::IsKeyDown(ImGuiKey_W))
            {
                _camera.position += _camera_move_speed * dt * _camera.front;
                camera_dirty = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_S))
            {
                _camera.position -= _camera_move_speed * dt * _camera.front;
                camera_dirty = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_A))
            {
                _camera.position -= _camera_move_speed * dt * _camera.right;
                camera_dirty = true;
            }
            if (ImGui::IsKeyDown(ImGuiKey_D))
            {
                _camera.position += _camera_move_speed * dt * _camera.right;
                camera_dirty = true;
            }
        }

        if (camera_dirty)
        {
            _camera = get_lookat_cam(_camera.position, _camera.position + _camera.front, _camera.up);
        }

        auto viewport = ImGui::GetMainViewport();
        auto p_min    = ImVec2{ viewport->Pos.x, viewport->Pos.y };
        auto p_max    = p_min + size;

        ImGui::GetBackgroundDrawList()->AddImage(_framebuffer_handle, p_min, p_max, { 0.f, 1.f }, { 1.f, 0.f });

        ImGui::Begin("Control");
        {
            ImGui::SliderFloat("Camera Move Speed", &_camera_move_speed, 0.1f, 10.f, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Camera Rotate Speed", &_camera_rotate_speed, 0.1f, 10.f, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Camera FOV", &_camera.fov, 10.f, 150.f, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::ColorPicker3("Background", &_bg_color.x, ImGuiColorEditFlags_Float);
        }
        ImGui::End();
    });
}

} // namespace lcgs