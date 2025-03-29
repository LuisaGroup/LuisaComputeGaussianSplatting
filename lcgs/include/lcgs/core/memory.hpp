#pragma once
/**
 * @file memory.h
 * @brief The Sail Memory
 * @author sailing-innocent
 * @date 2025-01-11
 */

#include <functional>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/device.h>

namespace lcgs
{

template <typename T>
void soa_obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
{
    std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1); // offset = minimum k * alignment > chunk
    ptr = reinterpret_cast<T*>(offset);
    chunk = reinterpret_cast<char*>(ptr + count);
}

template <typename T>
size_t soa_required(size_t N)
{
    char* size = nullptr;
    T::from_chunk(size, N);
    return ((size_t)size) + 128;
}

} // namespace lcgs