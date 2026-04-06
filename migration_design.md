# 并行原语迁移方案设计文档

## 1. 现状分析

### 1.1 当前项目并行原语实现 (自定义)

**位置**: `lcgs/include/lcgs/util/device_parallel.h`

**核心类**: `lcgs::DeviceParallel`

**功能**:
| 功能 | API |
|------|-----|
| 包含扫描 (Inclusive Sum) | `scan_inclusive_sum<T>(cmdlist, temp_buffer, d_in, d_out, init_v, num_item)` |
| 不包含扫描 (Exclusive Sum) | `scan_exclusive_sum<T>(cmdlist, temp_buffer, d_in, d_out, init_v, num_item)` |
| 归约 (Reduce) | `reduce<T>(cmdlist, temp_buffer, d_in, d_out, num_item, op)` |
| 基数排序 (Radix Sort) | `radix_sort<KeyType, ValueType>(...)` |

**实现特点**:
- 基于传统的前缀和算法（基于 CUDA SDK 的 prefix_sum.cu）
- 递归扫描实现
- 手动管理临时存储空间
- 模板特化实现类型支持

### 1.2 官方并行原语库 (lcpp)

**位置**: `../lc_parallel_primitive/src/lcpp/`

**核心类**:
| 类名 | 命名空间 | 功能 |
|------|----------|------|
| `DeviceScan` | `luisa::parallel_primitive` | 扫描操作（包含/不包含） |
| `DeviceReduce` | `luisa::parallel_primitive` | 归约操作 |
| `DeviceRadixSort` | `luisa::parallel_primitive` | 基数排序 |
| `DeviceFor` | `luisa::parallel_primitive` | 并行循环 |

**实现特点**:
- 先进的 One-Sweep 基数排序算法
- Decoupled Look-back 扫描算法
- 自动临时存储管理
- 支持更多操作类型（ArgMin, ArgMax, ReduceByKey 等）
- 更高效的线程协作原语（Warp/Block 级别）

## 2. 差异对比

### 2.1 API 接口差异

| 操作 | 当前实现 (lcgs) | 官方实现 (lcpp) | 差异说明 |
|------|-----------------|-----------------|----------|
| **包含扫描** | `scan_inclusive_sum<T>(cmdlist, temp, d_in, d_out, init_v, num_item)` | `InclusiveSum<T>(cmdlist, stream, d_in, d_out, num_items)` | lcpp需要stream参数，自动处理初始值 |
| **不包含扫描** | `scan_exclusive_sum<T>(cmdlist, temp, d_in, d_out, init_v, num_item)` | `ExclusiveSum<T>(cmdlist, stream, d_in, d_out, num_items)` | 同上 |
| **基数排序** | `radix_sort<K,V>(cmdlist, k_in, v_in, k_out, v_out, temp, num, bits)` | `SortPairs<K,V>(cmdlist, stream, k_in, k_out, v_in, v_out, num_items)` | lcpp自动管理临时存储，API更简洁 |
| **归约求和** | `reduce<T>(cmdlist, temp, d_in, d_out, num_item, op=0)` | `Sum<T>(cmdlist, stream, d_in, d_out, num_item)` | lcpp支持Sum/Min/Max/ArgMin/ArgMax等 |

### 2.2 临时存储管理

| 特性 | 当前实现 | 官方实现 |
|------|----------|----------|
| 临时存储 | 外部传入 BufferView | 内部自动创建/释放 |
| API 复杂度 | 需要两阶段调用 (先计算大小再执行) | 单阶段调用 |
| 内存效率 | 需要预分配最大空间 | 按需分配 |

### 2.3 性能对比

| 算法 | 当前实现 | 官方实现 |
|------|----------|----------|
| 扫描 | 递归扫描，多次 kernel 启动 | Decoupled Look-back，更少的 kernel 启动 |
| 排序 | 逐位基数排序 | One-Sweep 算法，单次遍历 |
| 吞吐量 | 中等 | 高 (接近 CUB 性能) |

## 3. 迁移方案

### 3.1 方案概述

**目标**: 将 `lcgs::DeviceParallel` 的实现从自定义版本替换为基于 `lcpp` 的封装。

**策略**: 
1. 创建适配器层 (`lcgs::DeviceParallel` 保持接口不变)
2. 内部实现转发到 `lcpp` 对应的类
3. 逐步替换使用方代码

### 3.2 适配器设计

```cpp
// 新的 device_parallel.h 结构

#include <lcpp/parallel_primitive.h>

namespace lcgs {

class DeviceParallel : public LuisaModule {
    // 内部使用 lcpp 的实现
    luisa::parallel_primitive::DeviceScan<> m_scan;
    luisa::parallel_primitive::DeviceRadixSort<> m_radix_sort;
    luisa::parallel_primitive::DeviceReduce<> m_reduce;
    
public:
    void create(Device& device);
    
    // 保持原有API签名
    template <NumericT Type4Byte>
    void scan_inclusive_sum(CommandList& cmdlist, 
                           BufferView<Type4Byte> temp_buffer,  // 保持兼容性，实际不使用
                           BufferView<Type4Byte> d_in,
                           BufferView<Type4Byte> d_out,
                           Type4Byte init_v,
                           size_t num_item);
    
    // 其他方法...
};

}
```

### 3.3 接口适配映射

| 原 API | 新 API 实现 |
|--------|-------------|
| `scan_inclusive_sum<T>(cmdlist, temp, d_in, d_out, init_v, num)` | `m_scan.InclusiveSum<T>(cmdlist, stream, d_in, d_out, num)` |
| `scan_exclusive_sum<T>(cmdlist, temp, d_in, d_out, init_v, num)` | `m_scan.ExclusiveSum<T>(cmdlist, stream, d_in, d_out, num)` |
| `radix_sort<K,V>(cmdlist, k_in, v_in, k_out, v_out, temp, num, bits)` | `m_radix_sort.SortPairs<K,V>(cmdlist, stream, k_in, k_out, v_in, v_out, num)` |
| `reduce<T>(cmdlist, temp, d_in, d_out, num, op)` | `m_reduce.Sum<T>(cmdlist, stream, d_in, d_out, num)` 或根据 op 选择 |

### 3.4 临时存储兼容性处理

由于原 API 需要外部传入 `temp_buffer`，但 lcpp 自动管理内存，处理方式：

```cpp
// 方案1: 保持原API签名但忽略temp_buffer参数
template <NumericT Type4Byte>
void scan_inclusive_sum(CommandList& cmdlist,
                       BufferView<Type4Byte> temp_buffer,  // 保持兼容，实际忽略
                       BufferView<Type4Byte> d_in,
                       BufferView<Type4Byte> d_out,
                       Type4Byte init_v,
                       size_t num_item) {
    // 创建临时stream用于同步（如果需要）
    // 或者从外部传入stream
    m_scan.InclusiveSum(cmdlist, m_stream, d_in, d_out, num_item);
}
```

## 4. 构建系统修改

### 4.1 xmake.lua 修改

```lua
-- lcgs/xmake.lua
target("lcgs")
    _config_project({
        project_kind = "shared"
    })
    add_deps("lc-core", "lc-runtime", "lc-vstl")
    add_deps("lc-dsl", "lc-backends-dummy")
    
    -- 新增: 依赖官方并行原语库
    add_deps("lcpp")
    add_includedirs("../lc_parallel_primitive/src", { public = false })
    
    add_defines("LCGS_DLL_EXPORTS")
    add_includedirs("include", { public = true })
    add_headerfiles("include/**.h", "include/**.hpp")
    set_pcxxheader("src/__pch.h")
    add_files("src/**.cpp")

    if is_host("windows") then
        add_syslinks("Advapi32", "User32", "d3d12", "Shell32")
    end
target_end()
```

## 5. 实施步骤

### 阶段1: 依赖集成
1. 修改 `lcgs/xmake.lua` 添加对 `lcpp` 的依赖
2. 验证构建系统可以正确找到 lcpp 头文件

### 阶段2: 适配器实现
1. 重写 `device_parallel.h` 使用 lcpp 内部实现
2. 保持原 API 签名以兼容现有代码
3. 实现所有必要的转发方法

### 阶段3: 验证测试
1. 编译项目确保没有编译错误
2. 运行现有测试验证功能正确性
3. 性能对比测试

### 阶段4: 代码优化 (可选)
1. 更新使用方代码直接使用 lcpp API
2. 移除不必要的临时存储参数
3. 简化调用流程

## 6. 风险与注意事项

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| API 不兼容 | 编译失败 | 保持适配器层接口不变 |
| 性能回退 | 运行变慢 | 充分测试，必要时优化 |
| 临时存储管理变化 | 内存问题 | 仔细检查资源生命周期 |
| Stream 参数差异 | 同步问题 | 明确管理 stream 传递 |

## 7. 代码示例

### 7.1 修改后的 device_parallel.h

```cpp
#pragma once
#include "lcgs/config.h"
#include "lcgs/core/runtime.h"
#include <lcpp/parallel_primitive.h>

namespace lcgs {

template <typename T>
static constexpr bool is_numeric_v = std::is_integral_v<T> || std::is_floating_point_v<T>;
template <typename T>
concept NumericT = is_numeric_v<T>;

class LCGS_API DeviceParallel : public LuisaModule {
public:
    int m_block_size = 256;
    
    DeviceParallel() = default;
    virtual ~DeviceParallel() = default;
    
    void create(Device& device);
    
    // 保持与原 API 兼容
    template <NumericT Type4Byte>
    void scan_inclusive_sum(size_t& temp_storage_size,
                           BufferView<Type4Byte> d_in,
                           BufferView<Type4Byte> d_out,
                           Type4Byte init_v,
                           size_t num_item);
    
    template <NumericT Type4Byte>
    void scan_inclusive_sum(CommandList& cmdlist,
                           BufferView<uint32_t> temp_buffer,
                           BufferView<Type4Byte> d_in,
                           BufferView<Type4Byte> d_out,
                           Type4Byte init_v,
                           size_t num_item);
    
    // 其他方法...
    
private:
    luisa::parallel_primitive::DeviceScan<> m_scan;
    luisa::parallel_primitive::DeviceRadixSort<> m_radix_sort;
    // ...
};

} // namespace lcgs
```

### 7.2 修改后的调用方式

```cpp
// 原代码 (main.cpp)
dp.scan_inclusive_sum<uint>(
    temp_space_size,
    d_tiles_touched,
    d_points_offset, 0, P
);
// ...
dp.radix_sort<luisa::ulong, luisa::uint>(
    sort_temp_size,
    d_point_list_keys_unsorted,
    d_point_list_unsorted,
    d_point_list_keys,
    d_point_list,
    L, 64
);

// 新代码 (需要传递 stream 参数)
// 或者通过适配器层隐藏 stream 参数
```

## 8. 时间估算

| 阶段 | 预计时间 |
|------|----------|
| 依赖集成 | 0.5 天 |
| 适配器实现 | 1-2 天 |
| 测试验证 | 1 天 |
| **总计** | **2.5-3.5 天** |
