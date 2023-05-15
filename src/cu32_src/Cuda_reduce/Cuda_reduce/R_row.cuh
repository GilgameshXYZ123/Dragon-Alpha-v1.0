#pragma once

#ifndef ROW_REDUCE_H
#define ROW_REDUCE_H

#include "row_linear.cuh"
#include "row_linear_dual.cuh"
#include "row_quadratic.cuh"
#include "row_quadratic_dual.cuh"
#include "row_linear_quadratic.cuh"
#include "row_min.cuh"
#include "row_max.cuh"
#include "row_min_indexed.cuh"
#include "row_max_indexed.cuh"
#include "row_softmax.cuh"
#include "row_softmax_crossEntropy_stage1.cuh"

//V1: holdY(), Y is not changed
//V2: holdX(), X is not changed
#include "row_layernorm_deltaXp_v1.cuh"
#include "row_layernorm_deltaXp_v1_affined.cuh"
#include "row_layernorm_deltaXp_v2.cuh"
#include "row_layernorm_deltaXp_v2_affined.cuh"

#endif
