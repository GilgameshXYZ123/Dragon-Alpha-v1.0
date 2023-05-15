#pragma once

#ifndef FIELD_REDUCE_H
#define FIELD_REDUCE_H

//V1: holdY(), Y is not changed 
//V2: holdX(), X is not changed

#include "field_linear.cuh"
#include "field_linear_dual.cuh"
#include "field_quadratic.cuh"
#include "field_quadratic_dual.cuh"
#include "field_linear_quadratic.cuh"

#include "field_max.cuh"
#include "field_min.cuh"
#include "field_max_indexed.cuh"
#include "field_min_indexed.cuh"

//------[affine, sqBatchNorm, batchNorm]--------
#include "field_affine_deltaA_v1.cuh"
#include "field_affine_deltaAB_v1.cuh"
#include "field_affine_deltaAB_v2.cuh"

#include "field_sqBatchnorm_deltaA_v2.cuh"
#include "field_sqBatchNorm_deltaAB_v2.cuh"

#include "field_batchNorm_deltaA_v2.cuh"
#include "field_batchNorm_deltaAB_v2.cuh"

//------[layerNorm]-----------------------------
#include "field_layernorm_deltaA_v2.cuh"
#include "field_layernorm_deltaAB_v2.cuh"

#endif