/*
constants for DES algorithm on CPU (same as in device_constants.h but without __device__ and named changed to host_*** to avoid name conflict)
all data from https://billstclair.com/grabbe/des.htm (link from project description)
*/
#pragma once
#include <stdint.h>

extern const uint8_t host_PC1[56];
extern const uint8_t host_SHIFT[16];
extern const uint8_t host_PC2[48];
extern const uint8_t host_IP[64];
extern const uint8_t host_E_BIT[48];

extern const uint8_t* host_S[8];
extern const uint8_t host_S1[64];
extern const uint8_t host_S2[64];
extern const uint8_t host_S3[64];
extern const uint8_t host_S4[64];
extern const uint8_t host_S5[64];
extern const uint8_t host_S6[64];
extern const uint8_t host_S7[64];
extern const uint8_t host_S8[64];

extern const uint8_t host_P[32];
extern const uint8_t host_IP_INV[64];
