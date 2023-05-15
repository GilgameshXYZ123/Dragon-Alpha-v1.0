#pragma once

#ifndef SQRT_2D_H
#define SQRT_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SQRT_2D_CALL
#define SQRT_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sqrt2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	sqrt2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#define sqrt2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	sqrt2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef SQRT_2D_KERNEL
#define SQRT_2D_KERNEL

//Y = sqrt(alpha*X + beta)
//Y' = 0.5 * alpha / sqrt(alpha*X + beta)
//Y' = 0.5 * alpha / Y
//deltaX = Y' * deltaY = (0.5 * alpha / Y) * deltaY
//	= (0.5 * alpha) * (deltaY / Y)

__global__ void sqrt2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; simdLinear4(y, alpha, x, beta);
		y.x = sqrtf(y.x);
		y.y = sqrtf(y.y);
		y.z = sqrtf(y.z);
		y.w = sqrtf(y.w); 

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __sqrt2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sqrt2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	sqrt2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif