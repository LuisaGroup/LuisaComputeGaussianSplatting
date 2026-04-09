set_xmakever("3.0.5")
add_rules("mode.release", "mode.debug", "mode.releasedbg")
set_languages("c++20")
includes("lc_options.generated.lua")
-------------------------------
-- DEPENDENCIES ---------------
includes(lc_dir)
-- LCPP: must be included after LuisaCompute (depends on lc-runtime, lc-dsl)
includes(lcpp_dir)
-------------------------------
includes("lcgs") -- lcgs.dll 
includes("test") -- lcgs-test.exe 
includes("app") -- lcgs-app.exe 
