/*
constants for DES algorithm (same as in device_constants.h but without __device__ and named without to host_***)
all data from https://billstclair.com/grabbe/des.htm (link from project description)
*/
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

extern __constant__ uint8_t PC1[56];
extern __constant__ uint8_t SHIFT[16];
extern __constant__ uint8_t PC2[48];
extern __constant__ uint8_t IP[64];
extern __constant__ uint8_t E_BIT[48];

extern __constant__ uint8_t* S[8];
extern __constant__ uint8_t S1[64];
extern __constant__ uint8_t S2[64];
extern __constant__ uint8_t S3[64];
extern __constant__ uint8_t S4[64];
extern __constant__ uint8_t S5[64];
extern __constant__ uint8_t S6[64];
extern __constant__ uint8_t S7[64];
extern __constant__ uint8_t S8[64];

extern __constant__ uint8_t P[32];
extern __constant__ uint8_t IP_INV[64];
