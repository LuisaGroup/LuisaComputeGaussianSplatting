target("lcgs-app")
    set_kind("binary")
    add_deps("lcgs", "stb-image", "lc-gui")
    add_files("*.cpp")
    -- enable RTTI
    add_cxxflags("/GR")
target_end()