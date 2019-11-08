#pragma once
#include "cpu_constants.h"
#include <stdint.h>
#include <stdio.h>

uint64_t doDES(uint64_t key, uint64_t data);
uint32_t f(uint32_t data, uint64_t key);
