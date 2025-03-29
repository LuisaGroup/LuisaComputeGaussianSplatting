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
    for (auto i : vstd::ptr_range(argv + 1, argc - 1))
    {
        vstd::string      arg     = i;
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