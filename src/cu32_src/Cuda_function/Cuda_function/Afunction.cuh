#pragma once

#ifndef FUNCTION_H
#define FUNCTION_H

//type1-----------------------------------------
#include "type1_equal_abs2D.cuh"
#include "type1_equal_abs2D_int.cuh"
#include "type1_equal_abs2D_char.cuh"
#include "type1_linear_greater2D.cuh"
#include "type1_linear_greater_dual2D.cuh"

#include "type1_linear2D.cuh"
#include "type1_linear2D_char2float.cuh"
#include "type1_linear2D_float2char.cuh"
#include "type1_linear2D_int2float.cuh"
#include "type1_linear2D_float2int.cuh"
#include "type1_linear_dual2D.cuh"
#include "type1_linear_dual2D_row.cuh"
#include "type1_linear_dual2D_field.cuh"
#include "type1_linear_dual_out2D.cuh"

#include "type1_quadratic2D.cuh"
#include "type1_quadratic2D_deltaX.cuh"
#include "type1_quadratic_dual2D.cuh"
#include "type1_quadratic_dual2D_deltaX.cuh"
#include "type1_quadratic_dual2D_row.cuh"
#include "type1_quadratic_dual2D_field.cuh"
#include "type1_variance2D_f64.cuh"

#include "type1_rpl2D.cuh"
#include "type1_rpl2D_deltaX.cuh"
#include "type1_div2D.cuh"
#include "type1_div2D_deltaX.cuh"
#include "type1_div2D_row.cuh"
#include "type1_div2D_field.cuh"
#include "type1_add_div2D_row.cuh"
#include "type1_add_div2D_field.cuh"

//type2-----------------------------------------
#include "type2_sign2D.cuh"
#include "type2_ceil2D.cuh"
#include "type2_floor2D.cuh"
#include "type2_abs2D.cuh"
#include "type2_abs2D_deltaX.cuh"
#include "type2_zero_nan2D.cuh"
#include "type2_sqrt2D.cuh"
#include "type2_sqrt_quadratic_dual2D.cuh"

//type3-----------------------------------------
#include "type3_min2D.cuh"
#include "type3_max2D.cuh"
#include "type3_min_dual2D.cuh"
#include "type3_max_dual2D.cuh"
#include "type3_clip2D.cuh"

//type4-----------------------------------------
#include "type4_relu2D.cuh"
#include "type4_relu2D_deltaX_v1.cuh"
#include "type4_relu2D_deltaX_v2.cuh"
#include "type4_leakyRelu2D.cuh"
#include "type4_leakyRelu2D_deltaX_v1.cuh"
#include "type4_leakyRelu2D_deltaX_v2.cuh"
#include "type4_elu2D.cuh"
#include "type4_elu2D_deltaX_v1.cuh"
#include "type4_elu2D_deltaX_v2.cuh"
#include "type4_softplus2D.cuh"
#include "type4_softplus2D_deltaX_v1.cuh"
#include "type4_softplus2D_deltaX_v2.cuh"
#include "type4_gelu2D.cuh"
#include "type4_gelu2D_deltaX.cuh"

//type5-----------------------------------------
#include "type5_tanh2D.cuh"
#include "type5_tanh2D_deltaX_v1.cuh"
#include "type5_tanh2D_deltaX_v2.cuh"
#include "type5_sigmoid2D.cuh"
#include "type5_sigmoid2D_deltaX_v1.cuh"
#include "type5_sigmoid2D_deltaX_v2.cuh"
#include "type5_exp2D.cuh"
#include "type5_log2D.cuh"
#include "type5_log2D_deltaX.cuh"
#include "type5_softmax2D_deltaX.cuh"
#include "type5_logsoftmax2D.cuh"
#include "type5_logsoftmax2D_deltaX.cuh"

//type6-----------------------------------------
#include "type6_sin2D.cuh"
#include "type6_sin2D_deltaX.cuh"
#include "type6_halfsin2D.cuh"
#include "type6_halfsin2D_deltaX.cuh"
#include "type6_tan2D.cuh"
#include "type6_tan2D_deltaX.cuh"
#include "type6_arcsin2D.cuh"
#include "type6_arcsin2D_deltaX.cuh"
#include "type6_arctan2D.cuh"
#include "type6_arctan2D_deltaX.cuh"

//distance-----------------------------------------
#include "distance_L1_2D.cuh"
#include "distance_L1_2D_deltaYh.cuh"
#include "distance_L2_2D.cuh"
#include "distance_L2_2D_deltaYh.cuh"
#include "distance_smoothL1_2D.cuh"
#include "distance_smoothL1_2D_deltaYh.cuh"

#include "distance_binaryCrossEntropy2D.cuh"
#include "distance_binaryCrossEntropy2D_deltaYh.cuh"
#include "distance_sigmoid_binaryCrossEntropy2D.cuh"
#include "distance_sigmoid_binaryCrossEntropy2D_deltaX.cuh"
#include "distance_crossEntropy2D.cuh"
#include "distance_crossEntropy2D_deltaYh.cuh"
#include "distance_softmax_crossEntropy2D.cuh"
#include "distance_softmax_crossEntropy2D_deltaX.cuh"

//optimizer-----------------------------------------
#include "optimizer_momentum2D.cuh"
#include "optimizer_momentum2D_decay.cuh"

#include "optimizer_sgdmn2D.cuh"
#include "optimizer_sgdmn2D_decay.cuh"

#include "optimizer_rmsprop2D.cuh"
#include "optimizer_rmsprop2D_decay.cuh"

#include "optimizer_adam2D.cuh"
#include "optimizer_adam2D_decay.cuh"
#include "optimizer_adam2D_type2.cuh"

#include "optimizer_adamW2D.cuh"

#include "optimizer_adamax2D.cuh"
#include "optimizer_adamax2D_decay.cuh"

#include "optimizer_adamod2D.cuh"
#include "optimizer_adamod2D_decay.cuh"

#include "extra_onehot2D_row_char.cuh"
#include "extra_onehot2D_row_int.cuh"
#include "extra_pix2tensor2D.cuh"

#include "extra_affine2D_row.cuh"

//----batch_norm---------------------------------------
#include "extra_sqBatchNorm2D_row.cuh"
#include "extra_sqBatchNorm2D_row_deltaX_v1.cuh"
#include "extra_sqBatchNorm2D_row_deltaX_v2.cuh"
#include "extra_sqBatchNorm_affined2D_row.cuh"
#include "extra_sqBatchNorm_affined2D_row_deltaX_v1.cuh"
#include "extra_sqBatchNorm_affined2D_row_deltaX_v2.cuh"

#include "extra_batchNorm2D_row.cuh"
#include "extra_batchNorm2D_row_deltaX_v1.cuh"
#include "extra_batchNorm2D_row_deltaX_v2.cuh"
#include "extra_batchNorm_affined2D_row.cuh"
#include "extra_batchNorm_affined2D_row_deltaX_v1.cuh"
#include "extra_batchNorm_affined2D_row_deltaX_v2.cuh"

//----layer_norm---------------------------------------
#include "extra_layernorm2D_row.cuh"
#include "extra_layernorm2D_row_deltaX_v1.cuh"
#include "extra_layernorm2D_row_deltaX_v2.cuh"
#include "extra_layernorm_affined2D_row.cuh"
#include "extra_layernorm_affined2D_row_deltaX_v1.cuh"
#include "extra_layernorm_affined2D_row_deltaX_v2.cuh"

#endif