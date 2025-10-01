/**
 * @file test_radix_sort_alignment.cpp
 * @brief Test suite for radix sort buffer alignment
 * @author copilot
 * @date 2025-01-09
 */

#include "test_util.h"
#include "lcgs/util/device_parallel.h"

TEST_SUITE("radix_sort_alignment") {
    TEST_CASE("temp_size_scan_alignment") {
        lcgs::DeviceParallel dp;
        
        // Test with various sizes to ensure alignment
        std::vector<size_t> test_sizes = {100, 1000, 2693, 5000, 10000};
        
        for (size_t num_items : test_sizes) {
            size_t temp_size = 0;
            dp.get_temp_size_scan(temp_size, num_items);
            
            // Check that temp_size is aligned to 4 elements (16 bytes for uint32_t)
            CAPTURE(num_items);
            CAPTURE(temp_size);
            CHECK(temp_size % 4 == 0);
        }
    }
    
    TEST_CASE("radix_sort_buffer_size_alignment") {
        lcgs::DeviceParallel dp;
        
        size_t num_items = 2693;  // This previously caused offset 2776 which is not aligned
        size_t temp_buffer_size = 0;
        
        // Mock buffers for the API call
        using KeyType = uint64_t;
        using ValueType = uint32_t;
        
        dp.radix_sort<KeyType, ValueType>(
            temp_buffer_size,
            luisa::compute::BufferView<KeyType>{},
            luisa::compute::BufferView<ValueType>{},
            luisa::compute::BufferView<KeyType>{},
            luisa::compute::BufferView<ValueType>{},
            num_items, 64
        );
        
        // Get the scan temp size separately
        size_t scan_temp_size = 0;
        dp.get_temp_size_scan(scan_temp_size, num_items);
        
        CAPTURE(num_items);
        CAPTURE(scan_temp_size);
        CAPTURE(temp_buffer_size);
        
        // Verify that scan_temp_size is aligned to 4 (16 bytes)
        CHECK(scan_temp_size % 4 == 0);
        
        // Verify that the total buffer size is large enough
        CHECK(temp_buffer_size >= scan_temp_size + num_items + 1);
    }
}
