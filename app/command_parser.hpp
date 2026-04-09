#pragma once
#include <luisa/vstl/common.h>
#include <luisa/vstl/functional.h>
#include <luisa/core/logging.h>
inline bool parse_command(
vstd::HashMap<vstd::string, vstd::function<void(vstd::string_view)>> const& cmds,
int argc, char* argv[],
vstd::string_view invalid_arg_cmd)
{
    bool result      = true;
    auto invalid_arg = [&]() {
        if (!invalid_arg_cmd.empty())
            LUISA_WARNING("{}", invalid_arg_cmd);
        result = false;
    };
    for (int idx = 1; idx < argc; ++idx)
    {
        vstd::string      arg     = argv[idx];
        vstd::string_view kv_pair = arg;
        for (auto i : vstd::range(arg.size()))
        {
            if (arg[i] == '-')
                continue;
            else
            {
                kv_pair = vstd::string_view(arg.data() + i, arg.size() - i);
                break;
            }
        }
        if (kv_pair.empty() || kv_pair.size() == arg.size()) [[unlikely]]
        {
            invalid_arg();
        }
        else
        {
            vstd::string_view key = kv_pair;
            vstd::string_view value;
            for (auto i : vstd::range(kv_pair.size()))
            {
                if (kv_pair[i] == '=')
                {
                    key   = vstd::string_view(kv_pair.data(), i);
                    value = vstd::string_view(kv_pair.data() + i + 1, kv_pair.size() - i - 1);
                    break;
                }
            }
            // If no '=' was found and value is empty, try to consume the next argument as the value
            if (value.empty() && key == kv_pair && idx + 1 < argc)
            {
                vstd::string_view next_arg = argv[idx + 1];
                // Only consume if the next argument doesn't start with '-' (i.e., it's not another flag)
                // But allow paths starting with a drive letter like D:\...
                bool next_is_flag = !next_arg.empty() && next_arg[0] == '-';
                if (next_is_flag && next_arg.size() >= 2)
                {
                    // Check if it's actually a negative number or just a flag
                    // If second char is a digit, treat as value (negative number)
                    if (next_arg[1] >= '0' && next_arg[1] <= '9')
                        next_is_flag = false;
                }
                if (!next_is_flag)
                {
                    value = next_arg;
                    ++idx; // skip the next argument since we consumed it as a value
                }
            }
            auto iter = cmds.find(key);
            if (!iter) [[unlikely]]
            {
                invalid_arg();
            }
            else
            {
                iter.value()(value);
            }
        }
    }
    return result;
}
