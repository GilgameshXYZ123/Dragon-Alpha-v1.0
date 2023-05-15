#pragma once

#ifndef QUADRATIC_2D_H
#define QUADRATIC_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef QUADRATIC_2D_CALL
#define QUADRATIC_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define quadratic2d_k4(stream, LB, LT, X, alpha, beta, gamma, Y, lengthv, width, stride)\
	quadratic2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, beta, gamma, Y, lengthv, width, stride)

#define quadratic2d_k4_small(stream, X, alpha, beta, gamma, Y, lengthv, width, stride)\
	quadratic2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


#ifndef QUADRATIC_2D_KERNEL
#define QUADRATIC_2D_KERNEL

//Quadratic: Y = alpha * X^2 + beta * X + gamma
//Y' = 2 * alpha * X + beta
__global__ void quadratic2D_kernel_4(
	const float* __restrict__ X, 
	float alpha, float beta, float gamma,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; //Y = alpha * X^2 + beta * X + gamma
		y.x = alpha * (x.x * x.x) + beta * x.x + gamma;
		y.y = alpha * (x.y * x.y) + beta * x.y + gamma;
		y.z = alpha * (x.z * x.z) + beta * x.z + gamma;
		y.w = alpha * (x.w * x.w) + beta * x.w + gamma;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __quadratic2D(cudaStream_t stream,
	const float* X, 
	float alpha, float beta, float gamma,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { quadratic2d_k4_small(stream, X, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	quadratic2d_k4(stream, 5, 2, X, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif