-- Function to read .env file and return a table of key-value pairs
local function read_env_file(filepath)
    local env = {}
    for line in io.lines(filepath) do
        local key, value = line:match("^([^=]+)=(.*)$")
        if key and value then
            env[tostring(key)] = tostring(value)
        end
    end
    return env
end
-- Read the .env file
local env = read_env_file(".env")

lc_options = {
    lc_cuda_backend = true,
    lc_dx_backend = true,
    lc_vk_backend = true,
    lc_enable_dsl = true,
    lc_enable_gui = true,
    lc_metal_backend = false,
    lc_enable_mimalloc = true,
    lc_enable_custom_malloc = false,
    lc_enable_unity_build = true,
    lc_enable_simd = true,
    lc_enable_api = false,
    lc_enable_clangcxx = false,
    lc_enable_osl = false,
    lc_enable_ir = false,
    lc_enable_tests = false,
    lc_external_marl = false,
    lc_dx_cuda_interop = false,
    lc_backend_lto = false,
    lc_sdk_dir = "\"\"",
}

local file = io.open("lc_options.generated.lua", "w")
file:write("lc_dir = ")
file:write(env.LC_DIR)
file:write("\n")
file:write("lc_options = {\n")
for key, value in pairs(lc_options) do
    file:write("    " .. key .. " = " .. tostring(value) .. ",\n")
end
file:write("}\n")
file:close()