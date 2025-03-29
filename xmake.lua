set_xmakever("2.9.8")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")
includes("lc_options.generated.lua")
includes(lc_dir)
includes("lcgs") -- lcgs.dll 
includes("test") -- lcgs-test.exe 
includes("app") -- lcgs-app.exe 
