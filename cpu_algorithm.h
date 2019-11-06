#pragma once
#include "cpu_constants.h"
#include <stdint.h>
#include <x86intrin.h> //I assume that we are on x86 processor with gcc

uint64_t doDES(uint64_t key, uint64_t data);
