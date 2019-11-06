/*
constants for DES algorithm on CPU (same as in device_constants.h but without __device__ and named changed to host_*** to avoid name conflict)
all data from https://billstclair.com/grabbe/des.htm (link from project description)
I modified some tables (-1 to indexes)
*/
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>

extern uint8_t host_PC1[56];
extern uint8_t host_SHIFT[16];
extern uint8_t host_PC2[48];
extern uint8_t host_IP[64];
extern uint8_t host_E_BIT[48];

extern uint8_t* host_S[8];
extern uint8_t host_S1[64];
extern uint8_t host_S2[64];
extern uint8_t host_S3[64];
extern uint8_t host_S4[64];
extern uint8_t host_S5[64];
extern uint8_t host_S6[64];
extern uint8_t host_S7[64];
extern uint8_t host_S8[64];

extern uint8_t host_P[32];
extern uint8_t host_IP_INV[64];
