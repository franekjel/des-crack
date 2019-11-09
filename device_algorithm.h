#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

#include "device_constants.h"

uint64_t CUDACrackDES(uint64_t encrypted, uint64_t decrypted);
