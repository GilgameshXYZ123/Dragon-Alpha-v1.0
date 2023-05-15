#pragma once

#ifndef LAYER_NORM_AFFINED_2D_ROW_DELTAX_V2_H
#define LAYER_NORM_AFFINED_2D_ROW_DELTAX_V2_H


//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LAYER_NORM_AFFINED_2D_ROW_DELTAX_V2_CALL
#define LAYER_NORM_AFFINED_2D_ROW_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define layernorm_affined2d_row_deltaX_v2_k4(stream, LB, LT, deltaY, X, X_row_mean, X_row_square_mean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	layernorm_affined2D_row_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, X, X_row_mean, X_row_square_mean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#define layernorm_affined2d_row_deltaX_v2_k4_small(stream, deltaY, X, X_row_mean, X_row_square_mean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	layernorm_affined2D_row_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, X, X_row_mean, X_row_square_mean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef LAYER_NORM_AFFINED_2D_ROW_DELTAX_V2_KERNEL
#define LAYER_NORM_AFFINED_2D_ROW_DELTAX_V2_KERNEL

//N: batch_size, M: row_length
//[1] Y = A*(X - X_mean[N]) / X_std[N] + B
//[2] X_std[N] = sqrt(X_square_mean - X_mean^2 + eps)
//(1) deltaX1 = deltaY * d"Y"/d"X" = deltaY * (A / X_std)
//(2) deltaX2 = deltaY * d"Y"/d"X_mean" * d"X_mean"/d"X"
//    deltaX2 = { row_sum:  deltaY * A * (X*X_mean - X_square_mean - eps) / (X_std^3) } * (1 / M)
//(3) deltaX3 = deltaY * d"Y"/d"X_var" * d"X_var"/d"X"
//    deltaX3 = { row_sum: -0.5 * deltaY * A * (X - X_mean) / (X_std^3) } * (2*X / M)
//    deltaX3 = { row_sum: deltaY * * A (X_mean - X) / (X_std^3) } * (X / M)
//(4) deltaX = deltaX1 + deltaX2 + deltaX3
//    deltaX = (deltaY * A / X_std) + 
//             { row_sum: deltaY * A * (X*X_mean - X_square_mean - eps) / (X_std^3) } * (1 / M)
//             { row_sum: deltaY * A * (X_mean - X) / (X_std^3) } * (X / M) 
//    deltaX = (deltaY * A / X_std) + {
//                 { row_sum: deltaY * A * (X*X_mean - X_square_mean - eps) } + 
//                 { row_sum: deltaY * A * (X_mean - X) } * X 
//              }/(M * X_std^3)	
//let: 
//<1> delteXp1[N] = row_sum: deltaY * A * (X*X_mean - X_square_mean - eps)
//<2> deltaXp2[N] = row_sum: deltaY * A * (X - X_mean)
//(5) deltaX = (deltaY * A / X_std) + (deltaXp1 - deltaXp2*X)/(M * X_std^3)	
//let: X_rstd = 1 / X_std
//    deltaX = (deltaY * A * X_rstd) +  (deltaXp1 - deltaXp2*X) * (X_rstd^3 / M)
//    deltaX = X_rstd * { deltaY * A + (deltaXp1 - deltaXp2*X) * (X_rstd^2 / M) }
//let: G = (X_rstd^2 / M)
//    deltaX = X_rstd * { deltaY * A + (deltaXp1 - deltaXp2*X)*G }
//    deltaX = X_rstd * { deltaY * A + deltaXp1*G - deltaXp2*G*X }
//let: G1 = deltaXp1*G, G2 = deltaXp2*G
//    deltaX = X_rstd * (deltaY*A + G1 - G2*X)
//Step:
//<1> X_rstd = 1 / X_std
//<2> G = (X_rstd^2 / M)
//<3> G1 = deltaXp1*G, G2 = deltaXp2*G
//<4> deltaX = X_rstd * (deltaY*A + G1 - G2*X)

__global__ void layernorm_affined2D_row_deltaX_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_row_mean,
	const float* __restrict__ X_row_square_mean, float eps,
	const float* __restrict__ A,
	const float* __restrict__ deltaXp1,
	const float* __restrict__ deltaXp2, int row_lengthv,
	float* __restrict__ deltaX,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	int M = (row_lengthv / stride) * width;//M = row_length

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 x = *(float4*)(X + index4);

		int row_index4 = index4 / row_lengthv;
		int field_index4 = index4 - row_index4 * row_lengthv;//index4 % row_lengthv;

		float x_mean = X_row_mean[row_index4];
		float x_smean = X_row_square_mean[row_index4];
		float dxp1 = deltaXp1[row_index4];
		float dxp2 = deltaXp2[row_index4];

		float4 a = *(float4*)(A + field_index4);

		//<1> X_rstd = 1 / X_std
		//<2> G = (X_rstd^2 / M)
		//<3> G1 = deltaXp1*G, G2 = deltaXp2*G
		float x_rstd = rsqrtf(x_smean - x_mean * x_mean + eps);
		float G = (x_rstd * x_rstd) / M;
		float G1 = dxp1 * G, G2 = dxp2 * G;

		float4 dx;//<4>  deltaX = X_rstd * (deltaY*A + G1 - G2*X)
		dx.x = x_rstd * (dy.x * a.x + G1 - G2 * x.x);
		dx.y = x_rstd * (dy.y * a.y + G1 - G2 * x.y);
		dx.z = x_rstd * (dy.z * a.z + G1 - G2 * x.z);
		dx.w = x_rstd * (dy.w * a.w + G1 - G2 * x.w);

		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __layernorm_affined2D_row_deltaX_v2(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_row_mean,
	const float* X_row_square_mean, float eps,
	const float* A,
	const float* deltaXp1, const float* deltaXp2, int row_lengthv,
	float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { layernorm_affined2d_row_deltaX_v2_k4_small(stream, deltaY, X, X_row_mean, X_row_square_mean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride); return; }
	layernorm_affined2d_row_deltaX_v2_k4(stream, 5, 2, deltaY, X, X_row_mean, X_row_square_mean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride);
}

#endif
