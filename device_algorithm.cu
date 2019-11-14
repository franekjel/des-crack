#include "device_algorithm.h"

static __device__ __forceinline__ uint64_t permutate56To56(uint64_t data, const uint8_t perm_table[])
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

static __device__ __forceinline__ uint64_t permutate56To48(uint64_t data, const uint8_t perm_table[])
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

static __device__ __forceinline__ uint64_t permutate64To64(uint64_t data, const uint8_t perm_table[])
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

static __device__ __forceinline__ uint64_t permutate32To48(uint32_t data, const uint8_t perm_table[])
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

static __device__ __forceinline__ uint32_t permutate32To32(uint32_t data, const uint8_t perm_table[])
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
static __device__ __forceinline__ uint32_t rol28(uint32_t data, uint8_t i)
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

static __device__ uint32_t f(uint32_t data, uint64_t key)
{
    uint64_t E = permutate32To48(data, E_BIT);
    uint32_t re = 0;
    E = E ^ key;
#pragma unroll
    for (unsigned int i = 0; i < 8; i++) {
        uint64_t k = ((E & (1L << (5 + (i * 6)))) >> (4 + (i * 6))) | ((E & (1L << (i * 6))) >> (i * 6));
        k <<= 4;
        k += (E << (59 - (i * 6))) >> 60;
        re |= (S[7 - i][k] << (i * 4));
    }
    return permutate32To32(re, P);
}

static __device__ uint64_t CUDAdoDES(uint64_t key, uint64_t data)
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

uint64_t expand56To64(uint64_t key)
{
    uint64_t re = 0;
    for (int i = 0; i < 8; i++) {
        uint64_t k = (key << (57 - 7 * i) >> 57);
        re |= (k << ((i * 8) + 1));
    }
    return re;
}

static __device__ uint64_t FoundKey = (1L << 60);

//__launch_bounds__(256) -slows down ~20% (64s->75s)
__global__ void kernel(uint64_t encrypted, uint64_t decrypted, uint64_t k)
{
    if (FoundKey != (1L << 60))
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
    /*  One thread crack 2^8 (change last 8 bits of given key)
        256 threads/block 2^8
        Grid size       2^16
        2^56 - all possible keys
    */
    cudaFuncSetCacheConfig(kernel, cudaFuncCachePreferL1);
    unsigned long start = time(NULL);
    //printf("Start at: %ld\n", time(NULL));
    unsigned int nBlock = 1 << 16;
    for (uint64_t k = 0; k < (1L << 56); k += (1L << 32)) {
        kernel<<<nBlock, 256>>>(encrypted, decrypted, k);
        uint64_t key;
        cudaDeviceSynchronize();
        cudaMemcpyFromSymbol(&key, FoundKey, 8);
        printf("Grid %ld: %lds\n", k >> 32, time(NULL) - start);
        start = time(NULL);
        if (key != (1L << 60)) {
            return expand56To64(key);
        }
    }
    return 0;
}
