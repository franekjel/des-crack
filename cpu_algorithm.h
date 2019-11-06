#pragma once
#include "cpu_constants.h"
#include <stdint.h>

uint64_t doDES(uint64_t key, uint64_t msg);
uint64_t permutate64To64(uint64_t data, uint8_t perm_table[]);
