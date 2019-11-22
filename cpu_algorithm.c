#include "cpu_algorithm.h"

inline static uint64_t permutate56To56(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
#pragma GCC unroll 64
#pragma GCC ivdep
    for (int i = 0; i < 56; i++) {
        uint64_t v = (data & (1L << (56 - perm_table[i])));
        v >>= (56 - perm_table[i]);
        v <<= 55 - i;
        re |= v;
    }
    return re;
}

inline static uint64_t permutate56To48(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
#pragma GCC unroll 64
#pragma GCC ivdep
    for (int i = 0; i < 48; i++) {
        uint64_t v = (data & (1L << (56 - perm_table[i])));
        v >>= (56 - perm_table[i]);
        v <<= 47 - i;
        re |= v;
    }
    return re;
}

inline static uint64_t permutate64To64(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
#pragma GCC unroll 64
#pragma GCC ivdep
    for (int i = 0; i < 64; i++) {
        uint64_t v = (data & (1L << (64 - perm_table[i])));
        v >>= (64 - perm_table[i]);
        v <<= 63 - i;
        re |= v;
    }
    return re;
}

inline static uint64_t permutate32To48(uint32_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
#pragma GCC unroll 64
#pragma GCC ivdep
    for (int i = 0; i < 48; i++) {
        uint64_t v = (data & (1L << (32 - perm_table[i])));
        v >>= (32 - perm_table[i]);
        v <<= 47 - i;
        re |= v;
    }
    return re;
}

inline static uint32_t permutate32To32(uint32_t data, const uint8_t perm_table[])
{
    uint32_t re = 0;
#pragma GCC unroll 64
#pragma GCC ivdep
    for (int i = 0; i < 32; i++) {
        uint64_t v = (data & (1L << (32 - perm_table[i])));
        v >>= (32 - perm_table[i]);
        v <<= 31 - i;
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

static uint64_t expand56To64(uint64_t key)
{
    uint64_t re = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t k = (key << (57 - 7 * i) >> 57);
        re |= (k << ((i * 8) + 1));
    }
    return re;
}

static uint32_t f(uint32_t data, uint64_t key)
{
    uint64_t E = permutate32To48(data, host_E_BIT);
    uint32_t re = 0;
    E = E ^ key;
#pragma GCC unroll 64
    for (unsigned int i = 0; i < 8; i++) {
        uint64_t k = ((E & (1L << (5 + (i * 6)))) >> (4 + (i * 6))) | ((E & (1L << (i * 6))) >> (i * 6));
        k <<= 4;
        k += (E << (59 - (i * 6))) >> 60;
        re |= (host_S[7 - i][k] << (i * 4));
    }
    return permutate32To32(re, host_P);
}

uint64_t doDES(uint64_t key, uint64_t data)
{
    //1
    uint64_t K[17];
    K[0] = permutate56To56(key, host_PC1);
    uint32_t C[17];
    uint32_t D[17];
    C[0] = (K[0] & 0x00fffffff0000000L) >> 28;
    D[0] = K[0] & 0x0fffffff;
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
    L[0] = M >> 32;
    R[0] = M & 0xffffffff;
    for (int i = 1; i < 17; i++) {
        L[i] = R[i - 1];
        R[i] = L[i - 1] ^ f(R[i - 1], K[i]);
    }
    M = R[16];
    M <<= 32;
    M |= L[16];
    return permutate64To64(M, host_IP_INV);
}

uint64_t CPUCrackDES(uint64_t encrypted, uint64_t decrypted)
{
    printf("Usig CPU to crack DES. This may take a while...\n(To be honest this may take very long...)\n");
    for (uint64_t key = 0; key < (1ULL << 56); key++) {
        if (doDES(key, decrypted) == encrypted)
            return expand56To64(key);
    }
    return 1ULL << 60;
}
