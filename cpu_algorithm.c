#include "cpu_algorithm.h"

uint64_t doDES(uint64_t key, uint64_t msg)
{
    msg = permutate64To64(msg, host_IP);
}

uint64_t permutate64To56(uint64_t data, uint8_t perm_table[])
{
    uint64_t re = 0;
    for (int i = 0; i < 56; i++) {
        uint64_t v = (data & (1L << perm_table[i]));
        v >>= perm_table[i];
        v <<= i;
        re |= v;
    }
    return re;
}
