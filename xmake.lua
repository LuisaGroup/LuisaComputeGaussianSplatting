add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")

if is_plat("windows") then
    set_runtimes("MD")
end

add_requires("luisa-compute[cuda,gui]")

add_requires("imgui v1.92.1-docking", "stb")

target("lcgs")
    set_kind("shared")
    add_files("lcgs/src/**.cpp")
    add_includedirs("lcgs/include", {public = true})
    set_pcxxheader("lcgs/src/__pch.h")
    add_defines("LCGS_DLL_EXPORTS")

    add_packages("luisa-compute")

target("lcgs-app")
    set_kind("binary")
    add_files("app/**.cpp")
    add_deps("lcgs")

    add_packages("luisa-compute", "imgui", "stb")

    on_config(function (target)
        -- Context require a path to find backend shared libraries
        -- Context context{argv[1]};
        -- target:add("runargs", path.join(target:pkg("luisa-compute"):installdir(), "bin"))
        -- -- Use target:targetdir() path
        -- -- Context context{argv[0]};
        os.vcp(path.join(target:pkg("luisa-compute"):installdir(), "bin/*.dll"), target:targetdir())
    end)
