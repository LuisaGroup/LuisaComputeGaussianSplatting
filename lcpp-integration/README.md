# LCPP (LuisaCompute Parallel Primitives) Integration

## GitHub 地址

官方 LCPP 仓库: https://github.com/LuisaGroup/lc_parallel_primitive

## 集成说明

本项目使用 xmake 作为构建工具，集成了 LuisaCompute 官方并行原语库 (LCPP)。

### 目录结构

```
lcpp-integration/
├── xmake.lua              # 主构建配置
├── src/
│   └── main.cpp           # 示例代码
└── tests/
    └── lcpp_test.cpp      # 测试代码
```

### 构建方式

```bash
# 配置并构建
xmake f -m release
xmake

# 运行测试
xmake run lcpp-test
```

### 使用示例

```cpp
#include <lcpp/parallel_primitive.h>
using namespace luisa::parallel_primitive;

// 创建设备
Context ctx{argv[0]};
Device device = ctx.create_device("cuda");
Stream stream = device.create_stream();

// Device-level reduce
DeviceReduce<> device_reduce;
device_reduce.create(device);

Buffer<int> input = device.create_buffer<int>(1024);
Buffer<int> output = device.create_buffer<int>(1);

CommandList cmdlist;
device_reduce.Sum(cmdlist, stream, input, output, 1024);
stream << cmdlist.commit() << synchronize();
```

### 提供的并行原语

- **Thread Level**: ThreadReduce, ThreadScan
- **Warp Level**: WarpReduce, WarpScan, WarpExchange
- **Block Level**: BlockReduce, BlockScan, BlockLoad, BlockStore, BlockRadixRank, BlockDiscontinuity
- **Device Level**: DeviceReduce, DeviceScan, DeviceRadixSort, DeviceSegmentReduce, DeviceHistogram
