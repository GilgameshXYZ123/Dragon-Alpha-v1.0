#pragma once

#ifndef LEAKY_RELU_2D_DELTAX_V2_H
#define LEAKY_RELU_2D_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef LEAKY_RELU_2D_DELTAX_V2_CALL
#define LEAKY_RELU_2D_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define leakyRelu2d_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, k, lengthv, width, stride)\
	leakyRelu2D_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, k, lengthv, width, stride)

#define leakyRelu2d_deltaX_v2_k4_small(stream,  deltaX, deltaY, X, k, lengthv, width, stride)\
	leakyRelu2D_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, k, lengthv, width, stride)

#endif


#ifndef LEAKY_RELU_2D_DELTAX_V2_KERNEL
#define LEAKY_RELU_2D_DELTAX_V2_KERNEL

//Y' = 1 + (Y <= 0)*k
//[1] X  > 0: Y = X, Y' = 1
//[2] X <= 0: Y = k*X, Y' = k
//Y' = (X > 0) + (X <= 0) * k
//Y' = (X > 0) + (1 - (X > 0))*k
//Y' = (X > 0)*(1 - k) + k

__global__ void leakyRelu2D_deltaX_v2_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X, float k,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 x = *(float4*)(X + index4);

		float4 dx;//(X > 0)*(1 - k) + k
		dx.x = (x.x > 0)*(1.0f - k) + k;
		dx.y = (x.y > 0)*(1.0f - k) + k;
		dx.z = (x.z > 0)*(1.0f - k) + k;
		dx.w = (x.w > 0)*(1.0f - k) + k;

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __leakyRelu2D_deltaX_v2(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *X, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { leakyRelu2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, k, lengthv, width, stride); return; }
	leakyRelu2d_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, k, lengthv, width, stride);
}

#endif