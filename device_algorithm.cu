#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "device_constants.h"

__global__ void kernel()
{
}

uint64_t crackDES(uint64_t encrypted, uint64_t decrypted)
{
    printf("Beginning cracking...\n");
}
