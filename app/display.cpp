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
    , _camera_rotate_speed{ 1.f }
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
        // enable main window docking
        ImGui::DockSpaceOverViewport(0, nullptr, ImGuiDockNodeFlags_PassthruCentralNode);

        constexpr auto fov_min = 1.f;
        constexpr auto fov_max = 150.f;

        auto viewport     = ImGui::GetMainViewport();
        auto camera_dirty = false;
        if (!ImGui::GetIO().WantCaptureMouse)
        {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
            {
                auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
                auto fov = luisa::radians(_camera.fov);
                _camera.front += 4.f * _camera_rotate_speed * (delta.x / viewport->Size.x) * _camera.right * fov;
                _camera.front -= 4.f * _camera_rotate_speed * (delta.y / viewport->Size.y) * _camera.up * fov;
                camera_dirty = true;
                ImGui::ResetMouseDragDelta(ImGuiMouseButton_Left);
            }
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
            {
                auto p_now  = ImGui::GetMousePos() - ImVec2{ .5f, .5f } * viewport->Size;
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
            if (ImGui::GetIO().MouseWheel != 0.f)
            {
                auto delta = ImGui::GetIO().MouseWheel * _camera_rotate_speed * .1f;
                _camera.fov = std::clamp(_camera.fov * std::exp2(delta), fov_min, fov_max);
                camera_dirty = true;
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
            auto camera   = get_lookat_cam(_camera.position, _camera.position + _camera.front, _camera.up);
            _camera.front = camera.front;
            _camera.right = camera.right;
            _camera.up    = camera.up;
        }

        ImGui::SetNextWindowBgAlpha(.5f);
        ImGui::Begin("Control", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
        {
            ImGui::Text("FPS: %.1f", ImGui::GetIO().Framerate);
            ImGui::Text("Camera Position: (%.2f, %.2f, %.2f)", _camera.position.x, _camera.position.y, _camera.position.z);
            ImGui::Text("Camera Front: (%.2f, %.2f, %.2f)", _camera.front.x, _camera.front.y, _camera.front.z);
            ImGui::Text("Camera Up: (%.2f, %.2f, %.2f)", _camera.up.x, _camera.up.y, _camera.up.z);
            ImGui::SliderFloat("Camera FOV", &_camera.fov, fov_min, fov_max, "%.1f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Camera Move Speed", &_camera_move_speed, 0.1f, 10.f, "%.1f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
            ImGui::SliderFloat("Camera Rotate Speed", &_camera_rotate_speed, 0.1f, 10.f, "%.1f", ImGuiSliderFlags_Logarithmic | ImGuiSliderFlags_AlwaysClamp);
            ImGui::ColorPicker3("Background", &_bg_color.x, ImGuiColorEditFlags_Float);
        }
        ImGui::End();

        auto p_min = ImVec2{ viewport->Pos.x, viewport->Pos.y };
        auto p_max = p_min + viewport->Size;
        ImGui::GetBackgroundDrawList()->AddImage(_framebuffer_handle, p_min, p_max, ImVec2{ 0.f, 1.f }, ImVec2{ 1.f, 0.f });
    });
}

} // namespace lcgs
