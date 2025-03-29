/**
 * @file bit_helper.cpp
 * @brief The Implement of bit helper
 * @author sailing-innocent
 * @date 2025-01-11
 */

#include "lcgs/util/misc.h"

namespace lcgs
{
uint32_t get_higher_msb(uint32_t n)
{
    uint32_t msb  = sizeof(n) * 4;
    uint32_t step = msb;
    while (step > 1)
    {
        step /= 2;
        if (n >> msb)
            msb += step;
        else
            msb -= step;
    }
    if (n >> msb)
    {
        msb++;
    }
    return msb;
}
} // namespace lcgs