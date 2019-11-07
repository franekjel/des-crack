#include "cpu_algorithm.h"

inline static uint64_t permutate64To56(uint64_t data, const uint8_t perm_table[])
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
inline static uint64_t permutate56To48(uint64_t data, const uint8_t perm_table[])
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

inline static uint64_t permutate64To64(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;

    for (int i = 0; i < 64; i++) {
        uint64_t v = (data & (1L << perm_table[i]));
        v >>= perm_table[i];
        v <<= i;
        re |= v;
    }
    return re;
}

inline static uint64_t permutate32To48(uint32_t data, const uint8_t perm_table[])
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
    printf("Get key: %lx\n", key);
    printf("Get data: %lx\n", data);
    //1
    uint64_t K[17];
    K[0] = permutate64To56(key, host_PC1);
    uint32_t C[17];
    uint32_t D[17];
    C[0] = __bextr_u64(K[0], 0x1c00);
    D[0] = __bextr_u64(K[0], 0x1c1b);
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
    //2
    uint64_t M = permutate64To64(data, host_IP);
    uint32_t L[17];
    uint32_t R[17];
    L[0] = __bextr_u64(M, 0x2000);
    R[0] = __bextr_u64(M, 0x201f);
    for (int i = 1; i < 17; i++) {
        L[i] = R[i - 1];
        R[i] = L[i - 1] ^ f(R[i - 1], K[i]);
    }
    M = R[16];
    M <<= 32;
    M |= L[16];

    printf("Encrypted to: %lx\n", permutate64To64(M, host_IP_INV));
    return permutate64To64(M, host_IP_INV);
}

uint32_t f(uint32_t data, uint64_t key)
{
    uint64_t E = permutate32To48(data, host_E_BIT);
    uint32_t re = 0;
    E = E ^ key;
#pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        uint64_t k = ((E & (1 << (5 + i * 6))) | ((E & (1 + i * 6)) << 4));
        k -= 1;
        k += __bextr_u64(E, 0x0400 | 1 + (i * 6)); //get 4 bits beginning from 1+(i*6)
        re |= host_S[i][k];
        re <<= 4;
    }
    return re;
}
