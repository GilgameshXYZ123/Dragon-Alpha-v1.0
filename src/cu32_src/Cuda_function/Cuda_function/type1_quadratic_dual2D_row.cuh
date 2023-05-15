#pragma once

#ifndef QUADRATIC_DUAL_2D_ROW_H
#define QUADRATIC_DUAL_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lengthv, row_lengthv] % stride == 0
//X2 must a 1D Tensor[row_lengthv]
//field_length * row_lengthv = X1.lengthv
#ifndef QUADRATIC_DUAL_2D_ROW_CALL
#define QUADRATIC_DUAL_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define quadratic_dual2d_row_k4_small(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

#define quadratic_dual2d_row_k4(stream, LB, LT, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

#endif


#ifndef QUADRATIC_DUAL_2D_ROW_KERNEL
#define QUADRATIC_DUAL_2D_ROW_KERNEL

//=========OLD fashion================================
//index4 % row_lengthv -> row_index
//x2.x = X2[(index4    ) % row_lengthv];
//x2.y = X2[(index4 + 1) % row_lengthv];
//x2.z = X2[(index4 + 2) % row_lengthv];
//x2.w = X2[(index4 + 3) % row_lengthv];
//=========OLD fashion================================

__global__ void quadratic_dual2D_row_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 y;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);

		int field_index4 = index4 % row_lengthv;//X2_lengthv % 4 == 0
		float4 x2 = *(float4*)(X2 + field_index4);

		//Y  = k11*X1^2 + k1*X1 = X1*(k11*X1 + k1)
		y.x = x1.x * (k11 * x1.x + k1);
		y.y = x1.y * (k11 * x1.y + k1);
		y.z = x1.z * (k11 * x1.z + k1);
		y.w = x1.w * (k11 * x1.w + k1);

		//Y += k11*X1^2 + k2*X2 = X2*(k22*X2 + k2)
		y.x += x2.x * (k22 * x2.x + k2);
		y.y += x2.y * (k22 * x2.y + k2);
		y.z += x2.z * (k22 * x2.z + k2);
		y.w += x2.w * (k22 * x2.w + k2);

		//Y += k22*X2^2 + C
		y.x += k12 * (x1.x * x2.x) + C;
		y.y += k12 * (x1.y * x2.y) + C;
		y.z += k12 * (x1.z * x2.z) + C;
		y.w += k12 * (x1.w * x2.w) + C;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __quadratic_dual2D_row(cudaStream_t stream,
	const float* X1,
	const float* X2, int row_lengthv,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { quadratic_dual2d_row_k4_small(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride); return; }
	quadratic_dual2d_row_k4(stream, 5, 2, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride);
}

#endif