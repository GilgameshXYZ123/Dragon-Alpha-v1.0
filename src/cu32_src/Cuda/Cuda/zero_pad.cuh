#pragma once

#ifndef ZERO_PAD_H
#define ZERO_PAD_H

struct char_3 { char a, b, c; };
struct char_4 { char a, b, c, d; };
struct __declspec(intrin_type) __declspec(align(16)) int_4 { int v[4]; };//128 bit
struct __declspec(intrin_type) __declspec(align(32)) int_8 { int v[8]; };//32 * 8 = 256 bit
struct __declspec(intrin_type) __declspec(align(64)) int_16 { int v[16]; };//32 * 8 = 256 bit

#include "zero_pad_funcs_w3s4.cuh"

#endif 
