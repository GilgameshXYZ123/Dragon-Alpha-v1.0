#pragma once

#ifndef LAYER_NORM_AFFINED_2D_ROW_DELTAX_V1_H
#define LAYER_NORM_AFFINED_2D_ROW_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LAYER_NORM_AFFINED_2D_ROW_DELTAX_V1_CALL
#define LAYER_NORM_AFFINED_2D_ROW_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define layernorm_affined2d_row_deltaX_v1_k4(stream, LB, LT, deltaY, Y, X_field_mean, X_field_square_mean, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	layernorm_affined2D_row_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, Y, X_field_mean, X_field_square_mean, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#define layernorm_affined2d_row_deltaX_v1_k4_small(stream, deltaY, Y, X_field_mean, X_field_square_mean, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	layernorm_affined2D_row_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, Y, X_field_mean, X_field_square_mean, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef LAYER_NORM_AFFINED_2D_ROW_DELTAX_V1_KERNEL
#define LAYER_NORM_AFFINED_2D_ROW_DELTAX_V1_KERNEL

//N: batch_size, M: row_lengthv
//[1] Y[N, M] = A[M] * (X - X_mean[N]) / X_std[N] + B[M]
//[2] deltaX = deltaX1 + deltaX2 + deltaX3
//(1) deltaX = deltaX1 + deltaX2 + deltaX3
//    deltaX = (deltaY * A / X_std) + {
//                 { row_sum: deltaY * A * (X*X_mean - X_square_mean - eps) } + 
//                 { row_sum: deltaY * A * (X_mean - X) } * X 
//              }/(M * X_std^3)	
//As: X = [(Y - B) / A] * X_std + X_mean
//(2) deltaX = (deltaY * A / X_std) + {
//                 { row_sum: deltaY * A * ([(Y - B) / A]*X_std*X_mean - X_std^2) } + 
//                 { row_sum: deltaY * A * [(B - Y) / A]*X_std } * X 
//              }/(M * X_std^3)	
//    deltaX = (deltaY * A / X_std) + {
//                 { row_sum: deltaY * ((Y - B)*X_std*X_mean - A * X_std^2) } + 
//                 { row_sum: deltaY * (B - Y)*X_std } * X 
//              }/(M * X_std^3)	
//    deltaX = (deltaY * A / X_std) + {
//                 { row_sum: deltaY * ((Y - B)*X_mean - A*X_std) } + 
//                 { row_sum: deltaY * (B - Y) } * X 
//              }/(M * X_std^2)	
//
//let: 
//<1> delteXp1[N] = row_sum: deltaY * ((Y - B)*X_mean - A*X_std)
//<2> deltaXp2[N] = row_sum: deltaY * (Y - B)
//(5) deltaX = (deltaY * A / X_std) + (deltaXp1 - deltaXp2*X)/(M * X_std^2)	
//let: X_rstd = 1 / X_std
//    deltaX = (deltaY * A * X_rstd) +  (deltaXp1 - deltaXp2*X) * (X_rstd^2 / M)
//    deltaX = X_rstd * { deltaY * A + (deltaXp1 - deltaXp2*X) * (X_rstd / M) }
//let: G = (X_rstd / M)
//    deltaX = X_rstd * { deltaY * A + (deltaXp1 - deltaXp2*X)*G }
//    deltaX = X_rstd * { deltaY * A + deltaXp1*G - deltaXp2*G*X }
//let: G1 = deltaXp1*G, G2 = deltaXp2*G
//    deltaX = X_rstd * (deltaY*A + G1 - G2*X)
//Step:
//<1> X_rstd = 1 / X_std
//<2> G = (X_rstd / M)
//<3> G1 = deltaXp1*G, G2 = deltaXp2*G
//<4> X = [(Y - B) / (X_rstd * A) + X_mean
//<5> deltaX = X_rstd * (deltaY*A + G1 - G2*X)

__global__ void layernorm_affined2D_row_deltaX_v1_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ X_row_mean,
	const float* __restrict__ X_row_square_mean, float eps,
	const float* __restrict__ A,
	const float* __restrict__ B,
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
		int field_index4 = index4 - row_index4 * row_lengthv;//index4 % row_lengthv;

		float x_mean = X_row_mean[row_index4];
		float x_smean = X_row_square_mean[row_index4];
		float dxp1 = deltaXp1[row_index4];
		float dxp2 = deltaXp2[row_index4];

		float4 a = *(float4*)(A + field_index4);
		float4 b = *(float4*)(B + field_index4);

		//<1> X_rstd = 1 / X_std
		//<2> G = (X_rstd / M)
		//<3> G1 = deltaXp1*G, G2 = deltaXp2*G
		float x_rstd = rsqrtf(x_smean - x_mean * x_mean + eps);
		float G = x_rstd / M;
		float G1 = dxp1 * G, G2 = dxp2 * G;

		float4 x;//<4> X = [(Y - B) / (X_rstd * A) + X_mean
		x.x = (y.x - b.x) / (x_rstd * a.x) + x_mean;
		x.y = (y.y - b.y) / (x_rstd * a.y) + x_mean;
		x.z = (y.z - b.z) / (x_rstd * a.z) + x_mean;
		x.w = (y.w - b.w) / (x_rstd * a.w) + x_mean;

		float4 dx;//<5> deltaX = X_rstd * (deltaY*A + G1 - G2*X)
		dx.x = x_rstd * (dy.x * a.x + G1 - G2 * x.x);
		dx.y = x_rstd * (dy.y * a.y + G1 - G2 * x.y);
		dx.z = x_rstd * (dy.z * a.z + G1 - G2 * x.z);
		dx.w = x_rstd * (dy.w * a.w + G1 - G2 * x.w);

		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __layernorm_affined2D_row_deltaX_v1(cudaStream_t stream,
	const float* deltaY, const float* Y,
	const float* X_field_mean,
	const float* X_field_square_mean, float eps,
	const float* A, const float* B,
	const float* deltaXp1, const float* deltaXp2, int row_lengthv,
	float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { layernorm_affined2d_row_deltaX_v1_k4_small(stream, deltaY, Y, X_field_mean, X_field_square_mean, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride); return; }
	layernorm_affined2d_row_deltaX_v1_k4(stream, 5, 2, deltaY, Y, X_field_mean, X_field_square_mean, eps, A, B, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride);
}

#endif
