set_xmakever("3.0.0")
set_version("0.1.0")

add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")

-- Options
option("cuda", {description = "Enable CUDA backend", default = true, type = "boolean"})
option("vulkan", {description = "Enable Vulkan backend", default = false, type = "boolean"})
option("cpu", {description = "Enable CPU backend", default = false, type = "boolean"})
option("gui", {description = "Enable GUI support", default = false, type = "boolean"})

-- LuisaCompute Repository Configuration
-- 从 GitHub 自动获取 LuisaCompute
local LC_REPO = "https://github.com/LuisaGroup/LuisaCompute.git"
local LC_BRANCH = "next"

-- 检查本地是否存在 LuisaCompute
local LC_LOCAL_DIR = "LuisaCompute"

-- 定义 LC 选项
lc_options = {
    lc_enable_xir = true,
    lc_enable_tests = false,
    lc_enable_dsl = true,
    lc_enable_cuda = has_config("cuda"),
    lc_enable_vulkan = has_config("vulkan"),
    lc_enable_cpu = has_config("cpu"),
    lc_enable_gui = has_config("gui"),
    lc_enable_remote = false,
    lc_enable_rust = false,
}

-- 包含 LuisaCompute (优先使用本地目录，否则自动获取)
if os.exists(LC_LOCAL_DIR) then
    includes(LC_LOCAL_DIR)
else
    -- 使用 package 方式添加 LuisaCompute
    includes("xmake/package.lua")
    add_requires("luisa-compute", {
        configs = {
            cuda = has_config("cuda"),
            vulkan = has_config("vulkan"),
            cpu = has_config("cpu"),
            gui = has_config("gui"),
        }
    })
end

-- LCPP Header-only Library Target
target("lcpp")
    set_kind("headeronly")
    set_languages("c++20")
    
    -- LCPP 头文件路径
    add_headerfiles("src/(lcpp/**.h)", {public = true})
    add_includedirs("src/", {public = true})
    
    on_load(function(target)
        if not os.exists("LuisaCompute") then
            target:add("packages", "luisa-compute", {public = true})
        else
            target:add("deps", "lc-runtime", "lc-dsl", {public = true})
        end
    end)
target_end()

-- Example executable
target("lcpp-example")
    set_kind("binary")
    set_languages("c++20")
    add_files("src/main.cpp")
    add_deps("lcpp")
    
    on_config(function(target)
        -- 添加运行时参数以便 Context 找到后端库
        if not os.exists("LuisaCompute") then
            target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
        end
    end)
target_end()

-- Test executable
target("lcpp-test")
    set_kind("binary")
    set_languages("c++20")
    add_files("tests/lcpp_test.cpp")
    add_deps("lcpp")
    
    on_config(function(target)
        if not os.exists("LuisaCompute") then
            target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
        end
    end)
target_end()
