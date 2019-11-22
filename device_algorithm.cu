#include "device_algorithm.h"

//modified to handle 56bit key
__constant__ uint8_t PC1[56] = {
    50, 43, 36, 29, 22, 15, 8,
    1, 51, 44, 37, 30, 23, 16,
    9, 2, 52, 45, 38, 31, 24,
    17, 10, 3, 53, 46, 39, 32,
    56, 49, 42, 35, 28, 21, 14,
    7, 55, 48, 41, 34, 27, 20,
    13, 6, 54, 47, 40, 33, 26,
    19, 12, 5, 25, 18, 11, 4
};

__constant__ uint8_t SHIFT[16] = {
    1, 1, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 1
};

__constant__ uint8_t PC2[48] = {
    14, 17, 11, 24, 1, 5,
    3, 28, 15, 6, 21, 10,
    23, 19, 12, 4, 26, 8,
    16, 7, 27, 20, 13, 2,
    41, 52, 31, 37, 47, 55,
    30, 40, 51, 45, 33, 48,
    44, 49, 39, 56, 34, 53,
    46, 42, 50, 36, 29, 32
};

__constant__ uint8_t IP[64] = {
    58, 50, 42, 34, 26, 18, 10, 2,
    60, 52, 44, 36, 28, 20, 12, 4,
    62, 54, 46, 38, 30, 22, 14, 6,
    64, 56, 48, 40, 32, 24, 16, 8,
    57, 49, 41, 33, 25, 17, 9, 1,
    59, 51, 43, 35, 27, 19, 11, 3,
    61, 53, 45, 37, 29, 21, 13, 5,
    63, 55, 47, 39, 31, 23, 15, 7
};

__constant__ uint8_t E_BIT[48] = {
    32, 1, 2, 3, 4, 5,
    4, 5, 6, 7, 8, 9,
    8, 9, 10, 11, 12, 13,
    12, 13, 14, 15, 16, 17,
    16, 17, 18, 19, 20, 21,
    20, 21, 22, 23, 24, 25,
    24, 25, 26, 27, 28, 29,
    28, 29, 30, 31, 32, 1
};

__constant__ uint8_t S1[64] = {
    14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7,
    0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8,
    4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0,
    15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13
};

__constant__ uint8_t S2[64] = {
    15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10,
    3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5,
    0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15,
    13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9
};

__constant__ uint8_t S3[64] = {
    10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8,
    13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1,
    13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7,
    1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12
};

__constant__ uint8_t S4[64] = {
    7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15,
    13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9,
    10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4,
    3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14
};

__constant__ uint8_t S5[64] = {
    2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9,
    14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6,
    4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14,
    11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3
};

__constant__ uint8_t S6[64] = {
    12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11,
    10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8,
    9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6,
    4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13
};

__constant__ uint8_t S7[64] = {
    4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1,
    13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6,
    1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2,
    6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12
};

__constant__ uint8_t S8[64] = {
    13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7,
    1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2,
    7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8,
    2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11
};

__constant__ uint8_t P[32] = {
    16, 7, 20, 21,
    29, 12, 28, 17,
    1, 15, 23, 26,
    5, 18, 31, 10,
    2, 8, 24, 14,
    32, 27, 3, 9,
    19, 13, 30, 6,
    22, 11, 4, 25
};

__constant__ uint8_t IP_INV[64] = {
    40, 8, 48, 16, 56, 24, 64, 32,
    39, 7, 47, 15, 55, 23, 63, 31,
    38, 6, 46, 14, 54, 22, 62, 30,
    37, 5, 45, 13, 53, 21, 61, 29,
    36, 4, 44, 12, 52, 20, 60, 28,
    35, 3, 43, 11, 51, 19, 59, 27,
    34, 2, 42, 10, 50, 18, 58, 26,
    33, 1, 41, 9, 49, 17, 57, 25
};

__device__ __forceinline__ uint64_t bfe(uint64_t data, unsigned pos, unsigned len)
{
    unsigned long re;
    asm("bfe.u64 %0, %1, %2, %3;"
        : "=l"(re)
        : "l"(data), "r"(pos), "r"(len));
    return re;
}

__device__ __forceinline__ uint64_t permutate56To56(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
    for (int i = 0; i < 56; i++) {
        uint64_t v = (data & (1L << (56 - perm_table[i])));
        v >>= (56 - perm_table[i]);
        v <<= 55 - i;
        re |= v;
    }
    return re;
}

__device__ __forceinline__ uint64_t permutate56To48(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
    for (int i = 0; i < 48; i++) {
        uint64_t v = (data & (1L << (56 - perm_table[i])));
        v >>= (56 - perm_table[i]);
        v <<= 47 - i;
        re |= v;
    }
    return re;
}

__device__ __forceinline__ uint64_t permutate64To64(uint64_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
    for (int i = 0; i < 64; i++) {
        uint64_t v = (data & (1L << (64 - perm_table[i])));
        v >>= (64 - perm_table[i]);
        v <<= 63 - i;
        re |= v;
    }
    return re;
}

__device__ __forceinline__ uint64_t permutate32To48(uint32_t data, const uint8_t perm_table[])
{
    uint64_t re = 0;
    for (int i = 0; i < 48; i++) {
        uint64_t v = (data & (1L << (32 - perm_table[i])));
        v >>= (32 - perm_table[i]);
        v <<= 47 - i;
        re |= v;
    }
    return re;
}

__device__ __forceinline__ uint32_t permutate32To32(uint32_t data, const uint8_t perm_table[])
{
    uint32_t re = 0;
    for (int i = 0; i < 32; i++) {
        uint64_t v = (data & (1L << (32 - perm_table[i])));
        v >>= (32 - perm_table[i]);
        v <<= 31 - i;
        re |= v;
    }
    return re;
}

//roll 28bit variable left 1 or 2
__device__ __forceinline__ uint32_t rol28(uint32_t data, uint8_t i)
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

__device__ uint32_t f(uint32_t data, uint64_t key)
{
    uint64_t E = permutate32To48(data, E_BIT);
    uint32_t re = 0;
    E = E ^ key;

    uint64_t k = ((E & (1L << 5)) >> 4) | (E & (1));
    k <<= 4;
    k += bfe(E, 1, 4);
    re |= S8[k];

    k = ((E & (1L << 11)) >> 10) | ((E & (1L << 6)) >> 6);
    k <<= 4;
    k += bfe(E, 7, 4);
    re |= (S7[k] << (4));

    k = ((E & (1L << 17)) >> 16) | ((E & (1L << 12)) >> 12);
    k <<= 4;
    k += bfe(E, 13, 4);
    re |= (S6[k] << 8);

    k = ((E & (1L << 23)) >> 22) | ((E & (1L << 18)) >> 18);
    k <<= 4;
    k += bfe(E, 19, 4);
    re |= (S5[k] << 12);

    k = ((E & (1L << 29)) >> 28) | ((E & (1L << 24)) >> 24);
    k <<= 4;
    k += bfe(E, 25, 4);
    re |= (S4[k] << 16);

    k = ((E & (1L << 35)) >> 34) | ((E & (1L << 30)) >> 30);
    k <<= 4;
    k += bfe(E, 31, 4);
    re |= (S3[k] << 20);

    k = ((E & (1L << 41)) >> 40) | ((E & (1L << 36)) >> 36);
    k <<= 4;
    k += bfe(E, 37, 4);
    re |= (S2[k] << 24);

    k = ((E & (1L << 47)) >> 46) | ((E & (1L << 42)) >> 42);
    k <<= 4;
    k += bfe(E, 43, 4);
    re |= (S1[k] << 28);

    return permutate32To32(re, P);
}

__device__ uint64_t CUDAdoDES(uint64_t key, uint64_t data)
{
    //1
    uint64_t K[17];
    K[0] = permutate56To56(key, PC1);
    uint32_t C[17];
    uint32_t D[17];
    C[0] = (K[0] & 0x00fffffff0000000L) >> 28;
    D[0] = K[0] & 0x0fffffff;
    for (int i = 1; i < 17; i++) {
        C[i] = rol28(C[i - 1], SHIFT[i - 1]);
        D[i] = rol28(D[i - 1], SHIFT[i - 1]);
    }
    for (int i = 1; i < 17; i++) {
        uint64_t p = C[i];
        p <<= 28;
        p |= D[i];
        K[i] = permutate56To48(p, PC2);
    }
    //2
    uint64_t M = permutate64To64(data, IP);
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
    return permutate64To64(M, IP_INV);
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

__device__ uint64_t FoundKey = (1ULL << 60);

__global__ void __launch_bounds__(256) kernel(uint64_t encrypted, uint64_t decrypted, uint64_t k)
{
    if (FoundKey != (1ULL << 60))
        return;
    uint64_t key = k + blockIdx.x * (1 << 16); //each block check 2^16 keys
    key += threadIdx.x * (1 << 8); //each thread check 256 (2^8) keys
    for (int i = 0; i < 256; i++) {
        if (CUDAdoDES(key + i, decrypted) == encrypted) {
            FoundKey = key + i;
        }
    }
}

uint64_t CUDACrackDES(uint64_t encrypted, uint64_t decrypted)
{
    printf("Beginning cracking usign GPU...\n");
    /*  One thread crack:  2^8
        256 threads/block: 2^8
        Grid size:         2^16
        Loop:              2^24
        2^56 = all possible keys
    */

    cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
    unsigned long start = time(NULL);
    unsigned int nBlock = 1 << 16;
    cudaProfilerStart();
    for (uint64_t k = 0; k < (1L << 56); k += (1L << 32)) {
        kernel<<<nBlock, 256>>>(encrypted, decrypted, k);
        uint64_t key;
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(&key, FoundKey, 8);
        printf("Grid %ld: %lds\n", k >> 32, time(NULL) - start);
        start = time(NULL);
        if (key != (1L << 60)) {
            cudaProfilerStop();
            return expand56To64(key);
        }
    }
    cudaProfilerStop();
    return 0;
}
