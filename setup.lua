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
    cuda_backend = true,
    dx_backend = true,
    enable_dsl = true,
    enable_gui = true,
    vk_support = true,
    lc_xrepo_dir = env.LC_XREPO_DIR,
    cpu_backend = false,
    metal_backend = false,
    enable_mimalloc = false,
    enable_custom_malloc = false,
    enable_api = false,
    enable_clangcxx = false,
    enable_osl = false,
    enable_ir = false,
    enable_tests = false,
    external_marl = false,
    dx_cuda_interop = false,
    lc_backend_lto = false,
    sdk_dir = "\"\"",
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