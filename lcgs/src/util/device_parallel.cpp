/**
 * @file helper/device_parallel_api.cpp
 * @author sailing-innocent
 * @brief The device parallel api implementation
 * @date 2023-12-28
 */

#include "lcgs/util/device_parallel.h"
#include <cstdint>

// API

namespace lcgs
{
using namespace luisa;
using namespace luisa::compute;

void DeviceParallel::create(Device& device)
{
    // argument will not change after create
    int num_elements_per_block = m_block_size * 2;
    int extra_space            = num_elements_per_block / m_num_banks;
    m_shared_mem_size          = (num_elements_per_block + extra_space);
    compile<int>(device);
    compile<float>(device);
    compile<uint32_t>(device);
    m_created = true;
}

void DeviceParallel::get_temp_size_scan(size_t& temp_storage_size, size_t num_items)
{
    auto         block_size       = m_block_size;
    unsigned int max_num_elements = num_items;
    temp_storage_size             = 0;
    unsigned int num_elements     = max_num_elements; // input segment size
    int          level            = 0;
    do
    {
        // output segment size
        unsigned int num_blocks = imax(1, (int)ceil((float)num_elements / (2.f * block_size)));
        if (num_blocks > 1)
        {
            level++;
            temp_storage_size += num_blocks;
        }
        num_elements = num_blocks;
    } while (num_elements > 1);
    temp_storage_size += 1;
    
    // Align to 16 bytes (4 elements for uint32_t buffer)
    // This ensures that subview offsets are properly aligned
    temp_storage_size = (temp_storage_size + 3) & ~3;
}

} // namespace lcgs