#include "cpu_algorithm.h"

inline static uint64_t permutate64To56(uint64_t data, uint8_t perm_table[])
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
inline static uint64_t permutate56To48(uint64_t data, uint8_t perm_table[])
{
    uint64_t re = 0;

    for (int i = 0; i < 48; i++) {
        uint64_t v = (data & (1L << perm_table[i]));
        v >>= perm_table[i];
        v <<= i;
        re |= v;
    }
    return re;
}

//roll 28bit variable left 1 or 2
inline static uint32_t rol28(uint32_t data, uint8_t i)
{
    if (i == 1) {
        data <<= 1;
        data |= (data & (1L << 28)) >> 28;
        data &= 0x0fffffff;
    } else { //i==2
        data <<= 2;
        data |= (data & (1L << 28)) >> 28;
        data |= (data & (1L << 29)) >> 28;
        data &= 0x0fffffff;
    }
    return data;
}

uint64_t doDES(uint64_t key, uint64_t data)
{
    //1
    uint64_t K[17];
    K[0] = permutate64To56(key, host_PC1);
    uint32_t C[17];
    uint32_t D[17];
    C[0] = __bextr_u64(K[0], 0x1c00);
    D[0] = __bextr_u64(K[0], 0x1c1c);
    for (int i = 1; i < 17; i++) {
        C[i] = rol28(C[i - 1], host_SHIFT[i - 1]);
        D[i] = rol28(D[i - 1], host_SHIFT[i - 1]);
    }
    for (int i = 1; i < 17; i++) {
        uint64_t p = C[i];
        p <<= 28;
        p |= D[i];
        K[i] = permutate56To48(p, host_PC2);
    }
}
