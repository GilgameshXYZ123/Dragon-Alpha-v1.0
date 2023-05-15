#pragma once

#ifndef TAN_2D_H
#define TAN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef TAN_2D_CALL
#define TAN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define tan2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	tan2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#define tan2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	tan2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef TAN_2D_KERNEL
#define TAN_2D_KERNEL

//Y = tan(alpha*X + beta) =
//Y' = alpha * (1 + tan^2(alpha*X + beta))
//Y' = alpha * (1 + y^2) = alpha*y^2 + alpha
//Y' = quadratic(alpha, 0, alpha)
//deltaX = deltaY * (alpha*y^2 + alpha)
//
//tan(x) = sin(x) / cos(x)
//cot(x) = cos(x) / sin(x)
// = - sin(x + 0.5pi) / cos(x + 0.5pi)
// = sin(-x - 0.5pi) / cos(x + 0.5pi)
// = sin(-x - 0.5pi) / cos(-x - 0.5pi)
//x -> a*x + b
//cot(a, b, x) = sin(-a*x - b - 0.5pi) / cos(-a*x - b - 0.5pi)
//cot(a*x + b) = sin(-a*x + (-b - 0.5pi)) / cos(-a*x + (-b - 0.5pi))
//cot(a*x + b) = tan(-a*x + (-b - 0.5pi))
//a -> -a
//b -> -b - 0.5pi

__global__ void tan2D_kernel_4(
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

		float4 y;//Y = tan(alpha*X + beta)
		y.x = tanf(alpha*x.x + beta);
		y.y = tanf(alpha*x.y + beta); 
		y.z = tanf(alpha*x.z + beta);
		y.w = tanf(alpha*x.w + beta);

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __tan2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { tan2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	tan2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif