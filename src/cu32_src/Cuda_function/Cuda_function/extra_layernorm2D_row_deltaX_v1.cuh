#pragma once

#ifndef LAYER_NORM_2D_ROW_DELTAX_V1_H
#define LAYER_NORM_2D_ROW_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
//affined = false
#ifndef LAYER_NORM_2D_ROW_DELTAX_V1_CALL
#define LAYER_NORM_2D_ROW_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define layernorm2d_row_deltaX_v1_k4(stream, LB, LT, deltaY, Y, X_row_mean, X_row_square_mean, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	layernorm2D_row_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, Y, X_row_mean, X_row_square_mean, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#define layernorm2d_row_deltaX_v1_k4_small(stream, deltaY, Y, X_row_mean, X_row_square_mean, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	layernorm2D_row_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, Y, X_row_mean, X_row_square_mean, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef LAYER_NORM_2D_ROW_DELTAX_V1_KERNEL
#define LAYER_NORM_2D_ROW_DELTAX_V1_KERNEL

//N: batch_size, M: row_length
//[1] Y = (X - X_mean[N]) / X_std[N]
//[2] X_std[N] = sqrt(X_square_mean - X_mean^2 + eps)
//(1) deltaX = (deltaY / X_std) + 
//             { row_sum: deltaY * (X*X_mean - X_square_mean - eps) / (X_std^3) } * (1 / M)
//             { row_sum: deltaY * (X_mean - X) / (X_std^3) } * (X / M) 
//    deltaX = (deltaY / X_std) + {
//                 { row_sum: deltaY * (X*X_mean - X_square_mean - eps) } + 
//                 { row_sum: deltaY * (X_mean - X) } * X 
//              }/(M * X_std^3)	
//(2) As: X = Y*X_std + X_mean
//    deltaX = (deltaY / X_std) + {
//                 { row_sum: deltaY * (Y*X_std*X_mean - X_std^2) } + 
//                 { row_sum: -deltaY * Y * X_std } * X 
//              }/(M * X_std^3)	
//   deltaX = (deltaY / X_std) + {
//                 { row_sum: deltaY * (Y*X_mean - X_std) } + 
//                 { row_sum: -deltaY * Y } * X 
//              }/(M * X_std^2)	
//let: 
//<1> deltaXp1[N] = { row_sum: deltaY * (Y*X_mean - X_std) }
//<2> deltaXp2[N] = { row_sum: deltaY * Y }
//
//(3) deltaX = (deltaY / X_std) + (deltaXp1 - deltaXp2*X) / (M * X_std^2)	
//let: X_rstd = 1 / X_std
//    deltaX = deltaY * X_rstd + (deltaXp1 - deltaXp2*X) * (X_rstd^2 / M)
//    deltaX = X_rstd * { deltaY + (deltaXp1 - deltaXp2*X) * (X_rstd / M) }
//let: G = (X_rstd / M)
//    deltaX = X_rstd * { deltaY + (deltaXp1 - deltaXp2*X) * G }
//    deltaX = X_rstd * { deltaY + deltaXp1*G - deltaXp2*G*X }
//let: G1 = deltaXp1*G, G2 = deltaXp2*G
//    deltaX = X_rstd * (deltaY + G1 - G2*X)
//As: G2*X = G2 * (Y / X_rstd + X_mean)
//         = G2 * (Y / X_rstd) + G2 * X_mean
//         = deltaXp2 * (X_rstd / M) * (Y / X_rstd) + G2 * X_mean
//         = (deltaXp2 / M)*Y + G2*X_mean
//    deltaX = X_rstd * (deltaY + G1 - (deltaXp2 / M)*Y - G2*X_mean)
//    deltaX = X_rstd * (deltaY + (G1 - G2*X_mean) - (deltaXp2/M)*Y)
//let: H1 = (G1 - G2*X_mean), H2 = (deltaXp2 / M)
//     H1 = deltaXp1*G - deltaXp2*G*X_mean
//     H1 = (deltaXp1 - deltaXp2*X_mean)*G
//    deltaX = X_rstd * (deltaY + H1 - H2*Y)
//Step:
//<1> X_rstd = 1 / X_std = 1 / sqrt(X_square_mean - X_mean^2 + eps)
//<2> G = (X_rstd / M)
//<3> H1 = (deltaXp1 - deltaXp2*X_mean)*G
//<4> H2 = (deltaXp2 / M)
//<5> deltaX = X_rstd * (deltaY + H1 - H2*Y)

__global__ void layernorm2D_row_deltaX_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ X_row_mean,
	const float* __restrict__ X_row_square_mean, float eps,
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
		float4 y = *(float4*)(Y + index4);

		int row_index4 = index4 / row_lengthv;
		float x_mean = X_row_mean[row_index4];
		float x_smean = X_row_square_mean[row_index4];
		float dxp1 = deltaXp1[row_index4];
		float dxp2 = deltaXp2[row_index4];

		//<1> X_rstd = 1 / X_std = 1 / sqrt(X_square_mean - X_mean^2 + eps)
		//<2> H1 = (deltaXp1 - deltaXp2*X_mean) * (X_rstd / M);
		//<3> H2 = (deltaXp2 / M)
		float x_rstd = rsqrtf(x_smean - x_mean * x_mean + eps);
		float H1 = (dxp1 - dxp2 * x_mean) * (x_rstd / M);
		float H2 = dxp2 / M;

		float4 dx;//<4> deltaX = X_rstd * (deltaY + H1 - H2*Y)
		dx.x = x_rstd * (dy.x + H1 - H2 * y.x);
		dx.y = x_rstd * (dy.y + H1 - H2 * y.y);
		dx.z = x_rstd * (dy.z + H1 - H2 * y.z);
		dx.w = x_rstd * (dy.w + H1 - H2 * y.w);

		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __layernorm2D_row_deltaX_v1(cudaStream_t stream,
	const float* deltaY, const float* Y,
	const float* X_row_mean,
	const float* X_row_square_mean, float eps,
	const float* deltaXp1, const float* deltaXp2, int row_lengthv,
	float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { layernorm2d_row_deltaX_v1_k4_small(stream, deltaY, Y, X_row_mean, X_row_square_mean, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride); return; }
	layernorm2d_row_deltaX_v1_k4(stream, 5, 2, deltaY, Y, X_row_mean, X_row_square_mean, eps, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride);
}

#endif
