#pragma once

#ifndef LINEAR_DUAL_2D_FIELD_H
#define LINEAR_DUAL_2D_FIELD_H

//lengthv = height * stride
//stride = (width + 3)/4 * 4, stride >= 4, stride % 4 ==0
//[lengthv, row_lengthv] % stride == 0
//X2 must a 1D Tensor[field_length]
//field_length * row_lengthv = X1.lengthv
#ifndef LINEAR_DUAL_2D_FIELD_CALL
#define LINEAR_DUAL_2D_FIELD_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define linear_dual2d_field_k4_small(stream, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_field_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

#define linear_dual2d_field_k4(stream, LB, LT, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_field_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_DUAL_2D_FIELD_KERNEL
#define LINEAR_DUAL_2D_FIELD_KERNEL

//for each field[i]:
//	Y[i] = alpha*X1[i] + beta*X2 + gamma
__global__ void linear_dual2D_field_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float alpha, float beta, float gamma,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float x2 = X2[index4 / row_lengthv] * beta + gamma;

		float4 y;
		y.x = (alpha * x1.x) + x2;
		y.y = (alpha * x1.y) + x2;
		y.z = (alpha * x1.z) + x2;
		y.w = (alpha * x1.w) + x2;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear_dual2D_field(cudaStream_t stream,
	const float* X1,
	const float* X2, int row_lengthv,
	float alpha, float beta, float gamma,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_dual2d_field_k4_small(stream, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	linear_dual2d_field_k4(stream, 5, 2, X1, X2, row_lengthv, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif