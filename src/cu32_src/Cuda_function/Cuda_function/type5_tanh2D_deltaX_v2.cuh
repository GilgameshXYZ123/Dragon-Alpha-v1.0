#pragma once

#ifndef TANH_2D_DELTAX_V2_H
#define TANH_2D_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef TANH_2D_DELTAX_V2_CALL
#define TANH_2D_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define tanh2d_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, lengthv, width, stride)\
	tanh2D_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#define tanh2d_deltaX_v2_k4_small(stream,  deltaX, deltaY, X, lengthv, width, stride)\
	tanh2D_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#endif


#ifndef TANH_2D_DELTAX_V2_KERNEL
#define TANH_2D_DELTAX_V2_KERNEL

//<1> Y = tan(X) = 2/(1 + e^(-2*X)) - 1
//<2> Y'= 1 - Y^2
__global__ void tanh2D_deltaX_v2_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 x = *(float4*)(X + index4);

		float4 y;//VERSION2: Y = 2/(1 + e^(-2*X)) - 1
		y.x = 2.0f / (1.0f + expf(-2.0f * x.x)) - 1.0f;
		y.y = 2.0f / (1.0f + expf(-2.0f * x.y)) - 1.0f;
		y.z = 2.0f / (1.0f + expf(-2.0f * x.z)) - 1.0f;
		y.w = 2.0f / (1.0f + expf(-2.0f * x.w)) - 1.0f;

		float4 dx;//Y'= 1 - Y^2
		dx.x = 1.0f - (y.x * y.x);
		dx.y = 1.0f - (y.y * y.y);
		dx.z = 1.0f - (y.z * y.z);
		dx.w = 1.0f - (y.w * y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __tanh2D_deltaX_v2(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* X,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { tanh2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	tanh2d_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, lengthv, width, stride);
}

#endif