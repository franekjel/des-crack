#pragma once
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#include "cpu_constants.h"

uint64_t doDES(uint64_t key, uint64_t data);
uint64_t CPUCrackDES(uint64_t encrypted, uint64_t decrypted);
