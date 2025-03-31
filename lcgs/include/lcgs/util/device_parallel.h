#pragma once
/**
 * @file helper/device_parallel.h
 * @author sailing-innocent
 * @brief The device parallel
 * @date 2023-12-28
 */

#include "lcgs/config.h"
#include "lcgs/core/runtime.h"
#include "lcgs/core/concept.hpp" // for NumericT
#include "lcgs/util/misc.hpp"    // for is_power_of_two

namespace lcgs
{

class LCGS_API DeviceParallel : public LuisaModule
{
    using IntType   = int;   // 4 byte
    using FloatType = float; // 4 byte

    template <NumericT Type4Byte>
    using ReduceShaderT = Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int, int>;
    template <typename KeyType, typename ValueType>
    using RadixSortCntShader = Shader<1, Buffer<uint32_t>, Buffer<KeyType>, int>;
    template <typename KeyType, typename ValueType>
    using RadixSortAssignShader = Shader<1, Buffer<KeyType>, Buffer<ValueType>, Buffer<KeyType>, Buffer<ValueType>, Buffer<uint32_t>, int>;

public:
    int m_block_size = 256;
    int m_num_banks  = 32;
    // shared_mem_banks = 2 ^ log_mem_banks
    int m_log_mem_banks = 5;

private:
    size_t m_shared_mem_size = 0;
    bool   m_created         = false;

public:
    // lifecycle
    DeviceParallel()          = default;
    virtual ~DeviceParallel() = default;
    void create(Device& device);
    // compile radix sort shader

    template <typename KeyType, typename ValueType>
    void enable_radix_sort(Device& device)
    {
        using namespace luisa;
        using namespace luisa::compute;

        if (!m_created)
        {
            create(device);
        }

        auto shader_desc_str = get_key_value_shader_desc<KeyType, ValueType>();

        // check if the shader is already compiled
        if (ms_radix_sort_cnt_map.find(shader_desc_str) != ms_radix_sort_cnt_map.end() &&
            ms_radix_sort_assign_map.find(shader_desc_str) != ms_radix_sort_assign_map.end())
        {
            return;
        }

        luisa::unique_ptr<RadixSortCntShader<KeyType, ValueType>>
            rs_cnt_shader = nullptr;

        lazy_compile(
            device, rs_cnt_shader,
            [](Var<Buffer<uint32_t>> cnt_buf,
               Var<Buffer<KeyType>>  key_buf,
               Int                   bit) {
                luisa::compute::set_block_size(256);
                auto i = dispatch_id().x;
                auto N = dispatch_size().x;
                $if(i < N)
                {
                    Var<KeyType> key  = key_buf.read(i);
                    UInt         cbit = static_cast<UInt>(key >> bit) & 1u;
                    cnt_buf.write(i, 1u - cbit);
                    $if(i == N - 1) // special case, store the last cnt
                    {
                        cnt_buf.write(N, 1u - cbit);
                    };
                };
            }
        );
        ms_radix_sort_cnt_map.try_emplace(shader_desc_str, std::move(rs_cnt_shader));

        luisa::unique_ptr<RadixSortAssignShader<KeyType, ValueType>>
            rs_assign_shader = nullptr;
        lazy_compile(
            device, rs_assign_shader,
            [](
                Var<Buffer<KeyType>>   in_key_buf,
                Var<Buffer<ValueType>> in_value_buf,
                Var<Buffer<KeyType>>   out_key_buf,
                Var<Buffer<ValueType>> out_value_buf,
                Var<Buffer<uint32_t>>  scan_buf,
                Int                    bit
            ) {
                luisa::compute::set_block_size(256);
                auto i           = dispatch_id().x;
                auto N           = dispatch_size().x;
                auto n_max_zeros = scan_buf.read(N) + scan_buf.read(N - 1);
                $if(i < N)
                {
                    Var<KeyType>   key   = in_key_buf.read(i);
                    Var<ValueType> value = in_value_buf.read(i);
                    UInt           cbit  = UInt(key >> bit) & 1u;
                    // UInt idx = cbit ? scan_buf.read(i) : cnt_buf.read(i);
                    UInt f = scan_buf.read(i);
                    $if(cbit == 0u)
                    {
                        UInt idx = f;
                        out_key_buf.write(idx, key);
                        out_value_buf.write(idx, value);
                    }
                    $else
                    {
                        UInt idx = n_max_zeros + i - f;
                        out_key_buf.write(idx, key);
                        out_value_buf.write(idx, value);
                    };
                };
            }
        );
        ms_radix_sort_assign_map.try_emplace(shader_desc_str, std::move(rs_assign_shader));
    }

    // API

    // First Call, get the temp_storage_size
    template <NumericT Type4Byte>
    void scan_inclusive_sum(
        size_t&               temp_storage_size,
        BufferView<Type4Byte> d_in,
        BufferView<Type4Byte> d_out,
        Type4Byte             init_v,
        size_t                num_item
    )
    {
        get_temp_size_scan(temp_storage_size, num_item);
        temp_storage_size += 1; // for the last element
    }

    // Second Call, do the scan
    template <NumericT Type4Byte>
    void scan_inclusive_sum(
        CommandList&          cmdlist,
        BufferView<uint32_t>  temp_buffer,
        BufferView<Type4Byte> d_in,
        BufferView<Type4Byte> d_out,
        Type4Byte             init_v,
        size_t                num_item
    )
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, num_item);
        temp_storage_size += 1; // for the last element
        LUISA_ASSERT(temp_storage_size <= temp_buffer.size(), "temp_buffer size is not enough");

        // store the last element to temp_buffer
        BufferView<Type4Byte> last_elem_buf = temp_buffer.subview(temp_storage_size - 1, 1).as<Type4Byte>();
        cmdlist << last_elem_buf.copy_from(d_in.subview(num_item - 1, 1));

        BufferView<Type4Byte> temp_buf = temp_buffer.subview(0, temp_storage_size - 1).as<Type4Byte>();

        // First perform an exclusive scan (without the initial value)
        prescan_array_recursive<Type4Byte>(
            cmdlist,
            temp_buf, d_in, d_out,
            num_item, 0, 0
        );
        // copy (1,N-1) -> (0,N-2)

        std::string_view key = luisa::compute::Type::of<Type4Byte>()->description();

        // inclusive sum special postprocess
        auto ms_inclusive_spec_it  = ms_inclusive_spec_map.find(key);
        auto ms_inclusive_spec_ptr = reinterpret_cast<Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>>*>(&(*ms_inclusive_spec_it->second));
        cmdlist << (*ms_inclusive_spec_ptr)(d_out, last_elem_buf).dispatch(num_item);

        // apply the initial value
        auto ms_add_it  = ms_add_map.find(key);
        auto ms_add_ptr = reinterpret_cast<Shader<1, Buffer<Type4Byte>, Type4Byte>*>(&(*ms_add_it->second));
        cmdlist << (*ms_add_ptr)(d_out, init_v).dispatch(num_item);
    }

    // First Call, get the temp_storage_size
    template <NumericT Type4Byte>
    void scan_exclusive_sum(
        size_t&               temp_storage_size,
        BufferView<Type4Byte> d_in,
        BufferView<Type4Byte> d_out,
        Type4Byte             init_v,
        size_t                num_item
    )
    {
        get_temp_size_scan(temp_storage_size, num_item);
    }

    // Second Call, do the scan
    template <NumericT Type4Byte>
    void scan_exclusive_sum(
        CommandList&          cmdlist,
        BufferView<Type4Byte> temp_buffer,
        BufferView<Type4Byte> d_in,
        BufferView<Type4Byte> d_out,
        Type4Byte             init_v,
        size_t                num_item
    )
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, num_item);
        LUISA_ASSERT(temp_storage_size <= temp_buffer.size(), "temp_buffer size is not enough");
        // LUISA_INFO("temp_storage_size: {}", temp_storage_size);

        prescan_array_recursive<Type4Byte>(
            cmdlist,
            temp_buffer, d_in, d_out,
            num_item, 0, 0
        );
        // LUISA_INFO("scan_exclusive_sum done");
        // add for all // brute force
        std::string_view key        = luisa::compute::Type::of<Type4Byte>()->description();
        auto             ms_add_it  = ms_add_map.find(key);
        auto             ms_add_ptr = reinterpret_cast<Shader<1, Buffer<Type4Byte>, Type4Byte>*>(&(*ms_add_it->second));
        cmdlist << (*ms_add_ptr)(d_out, init_v).dispatch(num_item);
    }

    template <NumericT Type4Byte>
    void reduce(size_t& temp_storage_size, BufferView<Type4Byte> d_in, BufferView<Type4Byte> d_out, size_t num_item, int op = 0)
    {
        get_temp_size_scan(temp_storage_size, num_item);
    }
    template <NumericT Type4Byte>
    void reduce(
        CommandList&          cmdlist,
        BufferView<Type4Byte> temp_buffer,
        BufferView<Type4Byte> d_in,
        BufferView<Type4Byte> d_out,
        size_t num_item, int op = 0
    )
    {
        size_t temp_storage_size = 0;
        get_temp_size_scan(temp_storage_size, num_item);
        LUISA_ASSERT(temp_buffer.size() >= temp_storage_size, "Please resize the Temp Buffer.");
        reduce_array_recursive<Type4Byte>(cmdlist, temp_buffer, d_in, d_out, num_item, 0, 0, op);
    }

    template <typename KeyType, typename ValueType>
    void radix_sort(size_t& temp_buffer_size, BufferView<KeyType> in_key_buffer, BufferView<ValueType> in_value_buffer, BufferView<KeyType> out_key_buffer, BufferView<ValueType> out_value_buffer, size_t num_item, int bits)
    {
        size_t scan_temp_storage_size = 0;
        get_temp_size_scan(scan_temp_storage_size, num_item);
        temp_buffer_size = scan_temp_storage_size + num_item + 1;
    }

    template <typename KeyType, typename ValueType>
    void radix_sort(
        CommandList&          cmdlist,
        BufferView<KeyType>   in_key_buffer,
        BufferView<ValueType> in_value_buffer,
        BufferView<KeyType>   out_key_buffer,
        BufferView<ValueType> out_value_buffer,
        BufferView<uint32_t>  temp_buffer,
        size_t num_item, int bits
    )
    {
        size_t scan_temp_storage_size = 0;
        get_temp_size_scan(scan_temp_storage_size, num_item);
        // LUISA_INFO("temp_buffer.size(): {}", temp_buffer.size());
        // LUISA_INFO("scan_temp_storage_size: {}", scan_temp_storage_size);

        LUISA_ASSERT(temp_buffer.size() >= scan_temp_storage_size + num_item + 1, "Please resize the Temp Buffer.");
        radix_sort_impl<KeyType, ValueType>(cmdlist, in_key_buffer, in_value_buffer, out_key_buffer, out_value_buffer, temp_buffer, num_item, bits);
    }

private:
    template <NumericT Type4Byte>
    void compile(Device& device)
    {

        using namespace luisa;
        using namespace luisa::compute;
        const auto n_blocks = m_block_size;

        size_t shared_mem_size = m_shared_mem_size;
        // see scanRootToLeavesInt in prefix_sum.cu
        auto scan_root_to_leaves = [&](SmemTypePtr<Type4Byte>& s_data, Int stride, Int n) {
            Int thid       = Int(thread_id().x);
            Int blockDim_x = Int(block_size().x);

            // traverse down the tree building the scan in place
            Int d = def(1);
            $while(d <= blockDim_x)
            {
                stride >>= 1;
                sync_block();

                $if(thid < d)
                {
                    Int i  = (stride * 2) * thid;
                    Int ai = i + stride - 1;
                    Int bi = ai + stride;
                    ai += conflict_free_offset(ai);
                    bi += conflict_free_offset(bi);
                    Var<Type4Byte> t = (*s_data)[ai];
                    (*s_data)[ai]    = (*s_data)[bi]; // left child <- root
                    (*s_data)[bi] += t;               // right child <- root + left child
                };
                d = d << 1;
            };
        };

        auto clear_last_element = [&](Int storeSum, SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& g_blockSums, Int blockIndex) {
            Int thid = Int(thread_id().x);
            Int d    = Int(block_size().x);
            $if(thid == 0)
            {
                Int index = (d << 1) - 1;
                index += conflict_free_offset(index);
                $if(storeSum == 1)
                {
                    // write this block's total sum to the corresponding index in the blockSums array
                    g_blockSums.write(blockIndex, (*s_data)[index]);
                };
                (*s_data)[index] = Type4Byte(0); // zero the last element in the scan so it will propagate back to the front
            };
        };
        // see buildSum in prefix_sum.cu
        auto build_sum = [&](SmemTypePtr<Type4Byte>& s_data, Int n) -> Int {
            Int thid   = Int(thread_id().x);
            Int stride = def(1);

            // build the sum in place up the tree
            Int d = Int(block_size().x);
            $while(d > 0)
            {
                sync_block();
                $if(thid < d)
                {
                    Int i  = (stride * 2) * thid;
                    Int ai = i + stride - 1;
                    Int bi = ai + stride;
                    ai += conflict_free_offset(ai);
                    bi += conflict_free_offset(bi);
                    (*s_data)[bi] += (*s_data)[ai];
                };
                stride *= 2;
                d = d >> 1;
            };
            return stride;
        };

        auto prescan_block = [&](Int storeSum, SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& blockSums, Int blockIndex, Int n) {
            $if(blockIndex == 0) { blockIndex = Int(block_id().x); };
            Int stride = build_sum(s_data, n); // build the sum in place up the tree
            clear_last_element(storeSum, s_data, blockSums, blockIndex);
            scan_root_to_leaves(s_data, stride, n); // traverse down tree to build the scan
        };

        auto load_shared_chunk_from_mem =
            [&](Int                     isNP2,
                SmemTypePtr<Type4Byte>& s_data,
                BufferVar<Type4Byte>&   g_idata,
                Int n, Int& baseIndex,
                Int& ai, Int& bi, Int& mem_ai, Int& mem_bi,
                Int& bankOffsetA, Int& bankOffsetB) {
                Int threadIdx_x = Int(thread_id().x);
                Int blockIdx_x  = Int(block_id().x);
                Int blockDim_x  = Int(block_size().x);

                Int thid = threadIdx_x;
                mem_ai   = baseIndex + threadIdx_x;
                mem_bi   = mem_ai + blockDim_x;

                ai          = thid;
                bi          = thid + blockDim_x;
                bankOffsetA = conflict_free_offset(ai); // compute spacing to avoid bank conflicts
                bankOffsetB = conflict_free_offset(bi);

                Var<Type4Byte> data_ai = Type4Byte(0);
                Var<Type4Byte> data_bi = Type4Byte(0);

                $if(ai < n) { data_ai = g_idata.read(mem_ai); }; // Cache the computational window in shared memory pad values beyond n with zeros
                $if(bi < n) { data_bi = g_idata.read(mem_bi); };
                (*s_data)[ai + bankOffsetA] = data_ai;
                (*s_data)[bi + bankOffsetB] = data_bi;
            };

        auto load_shared_chunk_from_mem_op =
            [&](SmemTypePtr<Type4Byte>& s_data, BufferVar<Type4Byte>& g_idata,
                Int n, Int& baseIndex, Int op) {
                Int threadIdx_x = Int(thread_id().x);
                Int blockIdx_x  = Int(block_id().x);
                Int blockDim_x  = Int(block_size().x);

                Int thid   = threadIdx_x;
                Int mem_ai = baseIndex + threadIdx_x;
                Int mem_bi = mem_ai + blockDim_x;

                Int ai          = thid;
                Int bi          = thid + blockDim_x;
                Int bankOffsetA = conflict_free_offset(ai); // compute spacing to avoid bank conflicts
                Int bankOffsetB = conflict_free_offset(bi);

                Var<Type4Byte> initial;
                $if(op == 0) { initial = Type4Byte(0); }                             // sum
                $elif(op == 1) { initial = std::numeric_limits<Type4Byte>::min(); }  // max
                $elif(op == 2) { initial = std::numeric_limits<Type4Byte>::max(); }; // min

                Var<Type4Byte> data_ai = initial;
                Var<Type4Byte> data_bi = initial;

                $if(ai < n) { data_ai = g_idata.read(mem_ai); }; // Cache the computational window in shared memory pad values beyond n with zeros
                $if(bi < n) { data_bi = g_idata.read(mem_bi); };
                (*s_data)[ai + bankOffsetA] = data_ai;
                (*s_data)[bi + bankOffsetB] = data_bi;
            };

        auto build_op =
            [&](SmemTypePtr<Type4Byte>& s_data, Int n, Int op) {
                Int thid   = Int(thread_id().x);
                Int stride = def(1);

                // build the sum in place up the tree
                Int d = Int(block_size().x);
                $while(d > 0)
                {
                    sync_block();
                    $if(thid < d)
                    {
                        Int i  = (stride * 2) * thid;
                        Int ai = i + stride - 1;
                        Int bi = ai + stride;
                        ai += conflict_free_offset(ai);
                        bi += conflict_free_offset(bi);
                        $if(op == 0) { (*s_data)[bi] += (*s_data)[ai]; }
                        $elif(op == 1) { (*s_data)[bi] = max((*s_data)[bi], (*s_data)[ai]); }
                        $elif(op == 2) { (*s_data)[bi] = min((*s_data)[bi], (*s_data)[ai]); };
                    };
                    stride *= 2;
                    d = d >> 1;
                };
            };

        auto reduce_block =
            [&](
                SmemTypePtr<Type4Byte>& s_data,
                BufferVar<Type4Byte>&   block_sums,
                Int                     block_index,
                Int n, Int op
            ) {
                $if(block_index == 0) { block_index = Int(block_id().x); };
                build_op(s_data, n, op); // build the op in place up the tree
                clear_last_element(1, s_data, block_sums, block_index);
            };

        // see storeSharedChunkToMem in prefix_sum.cu
        auto store_shared_chunk_to_mem =
            [&](Int isNP2, BufferVar<Type4Byte>& g_odata,
                SmemTypePtr<Type4Byte>& s_data,
                Int n, Int ai, Int bi, Int mem_ai, Int mem_bi, Int bankOffsetA, Int bankOffsetB) {
                sync_block();
                // write results to global memory
                $if(ai < n) { g_odata.write(mem_ai, (*s_data)[ai + bankOffsetA]); };
                $if(bi < n) { g_odata.write(mem_bi, (*s_data)[bi + bankOffsetB]); };
            };

        luisa::string_view key = Type::of<Type4Byte>()->description();
        luisa::unique_ptr<
            Shader<1, int, int, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int>>
            ms_prescan = nullptr;

        lazy_compile(
            device, ms_prescan,
            [&](Int storeSum, Int isNP2, // bool actually
                BufferVar<Type4Byte> g_idata,
                BufferVar<Type4Byte> g_odata,
                BufferVar<Type4Byte> g_blockSums,
                Int n, Int blockIndex, Int baseIndex) {
                set_block_size(n_blocks);
                Int                    ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
                Int                    blockIdx_x = Int(block_id().x);
                Int                    blockDim_x = Int(block_size().x);
                SmemTypePtr<Type4Byte> s_dataInt  = new SmemType<Type4Byte>{ shared_mem_size };
                $if(baseIndex == 0) { baseIndex = blockIdx_x * (blockDim_x << 1); };
                load_shared_chunk_from_mem(isNP2, s_dataInt, g_idata, n, baseIndex, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
                prescan_block(storeSum, s_dataInt, g_blockSums, blockIndex, n);
                store_shared_chunk_to_mem(isNP2, g_odata, s_dataInt, n, ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB);
            }
        );
        ms_prescan_map.try_emplace(luisa::string{ key }, std::move(ms_prescan));

        luisa::unique_ptr<
            Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int>>
            ms_uniform_add = nullptr;

        // see uniformAddInt in prefix_sum.cu
        lazy_compile(
            device, ms_uniform_add,
            [&](BufferVar<Type4Byte> g_data,
                BufferVar<Type4Byte> uniforms,
                Int n, Int blockOffset, Int baseIndex) {
                set_block_size(n_blocks);

                luisa::compute::Shared<Type4Byte> uni{ 1 };
                Int                               threadIdx_x = Int(thread_id().x);
                Int                               blockIdx_x  = Int(block_id().x);
                Int                               blockDim_x  = Int(block_size().x);
                $if(threadIdx_x == 0) { uni[0] = uniforms.read(blockIdx_x + blockOffset); };
                Int address = (blockIdx_x * (blockDim_x << 1)) + baseIndex + threadIdx_x;

                sync_block();

                // note two adds per thread
                $if(threadIdx_x < n)
                {
                    g_data.atomic(address).fetch_add(uni[0]);
                    $if(threadIdx_x + blockDim_x < n)
                    {
                        g_data.atomic(address + blockDim_x).fetch_add(uni[0]);
                    };
                };
            }
        );

        ms_uniform_add_map.try_emplace(luisa::string{ key }, std::move(ms_uniform_add));

        luisa::unique_ptr<
            Shader<1, Buffer<Type4Byte>, Type4Byte>>
            ms_add = nullptr;
        lazy_compile(
            device, ms_add,
            [&](luisa::compute::BufferVar<Type4Byte> buf, luisa::compute::Var<Type4Byte> v) {
                set_block_size(n_blocks);
                auto idx = dispatch_id().x;
                auto val = buf.read(idx) + v;
                buf.write(idx, val);
            }
        );
        ms_add_map.try_emplace(
            luisa::string{ luisa::compute::Type::of<Type4Byte>()->description() },
            std::move(ms_add)
        );

        luisa::unique_ptr<
            Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>>>
            ms_inclusive_spec = nullptr;
        lazy_compile(
            device, ms_inclusive_spec,
            [&](luisa::compute::BufferVar<Type4Byte> buf, luisa::compute::BufferVar<Type4Byte> bv) {
                set_block_size(n_blocks);
                auto idx = dispatch_id().x;
                auto val = buf.read(idx);
                $if(idx > 0)
                {
                    buf.write(idx - 1, val); // move partial
                };
                $if(idx == dispatch_size().x - 1)
                {
                    auto v = bv.read(0);
                    buf.write(idx, val + v); // add the last one
                };
            }
        );
        ms_inclusive_spec_map.try_emplace(
            luisa::string{ luisa::compute::Type::of<Type4Byte>()->description() },
            std::move(ms_inclusive_spec)
        );

        luisa::unique_ptr<
            ReduceShaderT<Type4Byte>>
            ms_reduce = nullptr;
        lazy_compile(
            device, ms_reduce,
            [&](BufferVar<Type4Byte> g_idata, BufferVar<Type4Byte> g_block_sums, Int n, Int block_index, Int base_index, Int op) {
                set_block_size(n_blocks);
                Int ai, bi, mem_ai, mem_bi, bank_offset_a, bank_offset_b;
                Int block_id_x  = Int(block_id().x);
                Int block_dim_x = Int(block_size().x);

                SmemTypePtr<Type4Byte> s_data = new SmemType<Type4Byte>{ shared_mem_size };
                $if(base_index == 0)
                {
                    base_index = block_id_x * (block_dim_x << 1);
                };
                load_shared_chunk_from_mem_op(s_data, g_idata, n, base_index, op);
                reduce_block(s_data, g_block_sums, block_index, n, op);
            }
        );

        ms_reduce_map.try_emplace(
            luisa::string{ luisa::compute::Type::of<Type4Byte>()->description() },
            std::move(ms_reduce)
        );
    }

    void                       get_temp_size_scan(size_t& temp_storage_size, size_t num_item);
    inline luisa::compute::Int conflict_free_offset(luisa::compute::Int i)
    {
        return i >> m_log_mem_banks;
    }

    template <NumericT Type4Byte>
    void prescan_array_recursive(
        CommandList&          cmdlist,
        BufferView<Type4Byte> temp_storage,
        BufferView<Type4Byte> arr_in,
        BufferView<Type4Byte> arr_out,
        size_t num_elements, int offset, int level
    ) noexcept
    {
        // using namespace inno::math;
        int block_size = m_block_size;
        int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
        int num_threads;

        if (num_blocks > 1)
        {
            num_threads = block_size;
        }
        else if (is_power_of_two(num_elements))
        {
            num_threads = num_elements / 2;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * 2;
        int num_elements_last_block =
            num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block / 2);
        int np2_last_block         = 0;
        int shared_mem_last_block  = 0;

        if (num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if (!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }

        size_t                size_elements = temp_storage.size() - offset;
        BufferView<Type4Byte> temp_buffer_level =
            temp_storage.subview(offset, size_elements);

        // execute the scan
        auto key            = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_prescan_it  = ms_prescan_map.find(key);
        auto ms_prescan_ptr = reinterpret_cast<
            Shader<1, int, int, Buffer<Type4Byte>, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int>*>(&(*ms_prescan_it->second));

        auto ms_uniform_add_it  = ms_uniform_add_map.find(key);
        auto ms_uniform_add_ptr = reinterpret_cast<
            Shader<1, Buffer<Type4Byte>, Buffer<Type4Byte>, int, int, int>*>(&(*ms_uniform_add_it->second));

        if (num_blocks > 1)
        {
            // recursive
            cmdlist << (*ms_prescan_ptr)(true, false, arr_in, arr_out, temp_buffer_level, num_elements_per_block, 0, 0)
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if (np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_prescan_ptr)(
                               true, true,
                               arr_in, arr_out, temp_buffer_level,
                               num_elements_last_block, num_blocks - 1,
                               num_elements - num_elements_last_block
                )
                               .dispatch(block_size);
            }

            prescan_array_recursive<Type4Byte>(
                cmdlist,
                temp_buffer_level, temp_buffer_level,
                temp_buffer_level,
                num_blocks, num_blocks, level + 1
            );

            cmdlist << (*ms_uniform_add_ptr)(
                           arr_out, temp_buffer_level,
                           num_elements - num_elements_last_block,
                           0, 0
            )
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if (np2_last_block)
            {
                cmdlist << (*ms_uniform_add_ptr)(
                               arr_out, temp_buffer_level,
                               num_elements_last_block, num_blocks - 1,
                               num_elements - num_elements_last_block
                )
                               .dispatch(block_size);
            }
        }
        else if (is_power_of_two(num_elements))
        {
            // non-recursive
            cmdlist << (*ms_prescan_ptr)(
                           false, false,
                           arr_in, arr_out, temp_buffer_level,
                           num_elements, 0, 0
            )
                           .dispatch(block_size);
        }
        else
        {
            // non-recursive
            cmdlist << (*ms_prescan_ptr)(
                           false, true,
                           arr_in, arr_out, temp_buffer_level,
                           num_elements, 0, 0
            )
                           .dispatch(block_size);
        }
    }

    template <NumericT Type4Byte>
    void reduce_array_recursive(
        luisa::compute::CommandList& cmdlist,
        BufferView<Type4Byte>        temp_storage,
        BufferView<Type4Byte>        arr_in,
        BufferView<Type4Byte>        arr_out,
        int                          num_elements,
        int offset, int level, int op
    ) noexcept
    {
        int block_size = m_block_size;
        int num_blocks = imax(1, (int)ceil((float)num_elements / (2.0f * block_size)));
        int num_threads;

        if (num_blocks > 1)
        {
            num_threads = block_size;
        }
        else if (is_power_of_two(num_elements))
        {
            num_threads = num_elements / 2;
        }
        else
        {
            num_threads = floor_pow_2(num_elements);
        }

        int num_elements_per_block = num_threads * 2;
        int num_elements_last_block =
            num_elements - (num_blocks - 1) * num_elements_per_block;
        int num_threads_last_block = imax(1, num_elements_last_block / 2);
        int np2_last_block         = 0;
        int shared_mem_last_block  = 0;

        if (num_elements_last_block != num_elements_per_block)
        {
            // NOT POWER OF 2
            np2_last_block = 1;
            if (!is_power_of_two(num_elements_last_block))
            {
                num_threads_last_block = floor_pow_2(num_elements_last_block);
            }
        }

        size_t                size_elements = temp_storage.size() - offset;
        BufferView<Type4Byte> temp_buffer_level =
            temp_storage.subview(offset, size_elements);

        // execute the scan
        auto key           = luisa::compute::Type::of<Type4Byte>()->description();
        auto ms_reduce_it  = ms_reduce_map.find(key);
        auto ms_reduce_ptr = reinterpret_cast<
            ReduceShaderT<Type4Byte>*>(&(*ms_reduce_it->second));

        if (num_blocks > 1)
        {
            // recursive
            cmdlist << (*ms_reduce_ptr)(
                           arr_in, temp_buffer_level,
                           num_elements, 0, 0, op
            )
                           .dispatch(block_size * (num_blocks - np2_last_block));

            if (np2_last_block)
            {
                // Last Block
                cmdlist << (*ms_reduce_ptr)(
                               arr_in, temp_buffer_level,
                               num_elements_last_block, num_blocks - 1,
                               num_elements - num_elements_last_block, op
                )
                               .dispatch(block_size);
            }

            reduce_array_recursive<Type4Byte>(
                cmdlist,
                temp_buffer_level, temp_buffer_level,
                arr_out,
                num_blocks, num_blocks, level + 1, op
            );
        }
        else
        {
            // non-recursive
            cmdlist << (*ms_reduce_ptr)(
                           arr_in, temp_buffer_level,
                           num_elements, 0, 0, op
            )
                           .dispatch(block_size);
            cmdlist << arr_out.copy_from(temp_buffer_level);
        }
    }

    template <typename KeyType, typename ValueType>
    void radix_sort_impl(
        CommandList&          cmdlist,
        BufferView<KeyType>   in_key_buffer,
        BufferView<ValueType> in_value_buffer,
        BufferView<KeyType>   out_key_buffer,
        BufferView<ValueType> out_value_buffer,
        BufferView<uint32_t>  temp_buffer,
        size_t num_item, int bits
    )
    {
        auto desc = get_key_value_shader_desc<KeyType, ValueType>();

        auto ms_radix_sort_cnt_it = ms_radix_sort_cnt_map.find(desc);
        LUISA_ASSERT(ms_radix_sort_cnt_it != ms_radix_sort_cnt_map.cend(), "GET SHADER FAILED");
        auto ms_radix_sort_cnt_ptr = reinterpret_cast<
            RadixSortCntShader<KeyType, ValueType>*>(&(*ms_radix_sort_cnt_it->second));

        auto ms_radix_sort_assign_it = ms_radix_sort_assign_map.find(desc);
        LUISA_ASSERT(ms_radix_sort_assign_it != ms_radix_sort_assign_map.cend(), "GET SHADER FAILED");

        auto ms_radix_sort_assign_ptr = reinterpret_cast<
            RadixSortAssignShader<KeyType, ValueType>*>(&(*ms_radix_sort_assign_it->second));

        size_t scan_temp_size = 0;
        get_temp_size_scan(scan_temp_size, num_item);
        auto scan_temp_buffer = temp_buffer.subview(0, scan_temp_size);
        auto cnt_buffer       = temp_buffer.subview(scan_temp_size, num_item + 1);

        // here we use exclusive sum to implement the radix sort
        for (int bit = 0; bit < bits; ++bit)
        {
            cmdlist << (*ms_radix_sort_cnt_ptr)(cnt_buffer, in_key_buffer, bit).dispatch(num_item);
            scan_exclusive_sum<uint32_t>(cmdlist, scan_temp_buffer, cnt_buffer, cnt_buffer, 0, num_item);
            cmdlist << (*ms_radix_sort_assign_ptr)(in_key_buffer, in_value_buffer, out_key_buffer, out_value_buffer, cnt_buffer, bit).dispatch(num_item);
            cmdlist << in_key_buffer.copy_from(out_key_buffer);
            cmdlist << in_value_buffer.copy_from(out_value_buffer);
        }
    }

    template <typename KeyType, typename ValueType>
    luisa::string get_key_value_shader_desc()
    {
        luisa::string_view key_desc   = luisa::compute::Type::of<KeyType>()->description();
        luisa::string_view value_desc = luisa::compute::Type::of<ValueType>()->description();

        return luisa::string(key_desc) + "+" + luisa::string(value_desc);
    }

    // for scan
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_prescan_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_uniform_add_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_add_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_inclusive_spec_map;
    // for reduce
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_reduce_map;
    // for radix sort
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_cnt_map;
    luisa::unordered_map<luisa::string, luisa::shared_ptr<luisa::compute::Resource>> ms_radix_sort_assign_map;
};

} // namespace lcgs