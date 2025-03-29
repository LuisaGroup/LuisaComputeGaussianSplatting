target("lcgs")
    _config_project({
        project_kind = "shared"
    })
    add_deps("lc-core", "lc-runtime", "lc-vstl")
    add_deps("lc-dsl", "lc-ast", "lc-backends-dummy")
    add_defines("LCGS_DLL_EXPORT")

    add_includedirs("include", { public = true })
    add_headerfiles("include/**.h", "include/**.hpp")
    set_pcxxheader("src/__pch.h")
    add_files("src/**.cpp")

    if is_host("windows") then
        add_syslinks("Advapi32", "User32", "d3d12", "Shell32")
    end
target_end()