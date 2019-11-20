#include <cuda.h>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

u_int64_t CUDACrackDES(uint64_t encrypted, uint64_t decrypted);
