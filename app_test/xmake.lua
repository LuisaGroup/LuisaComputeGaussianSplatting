target("lcgs-app")
    _config_project({
        project_kind = "binary"
    })
    add_deps("lcgs", "stb-image", "lc-gui")
    add_files("*.cpp")
target_end()