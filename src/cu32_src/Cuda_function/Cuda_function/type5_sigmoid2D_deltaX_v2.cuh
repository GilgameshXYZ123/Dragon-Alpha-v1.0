#pragma once

#ifndef SIGMOID_2D_DELTAX_V2_H
#define SIGMOID_2D_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef SIGMOID_2D_DELTAX_V2_CALL
#define SIGMOID_2D_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sigmoid2d_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, lengthv, width, stride)\
	sigmoid2D_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#define sigmoid2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, lengthv, width, stride)\
	sigmoid2D_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#endif


#ifndef SIGMOID_2D_DELTAX_V2_KERNEL
#define SIGMOID_2D_DELTAX_V2_KERNEL

//Y = 1 / (1 + e^(-X))
//Y' = exp(X) / (1 + exp(X))^2
//Step: 
//<1> Y = 1 / (1 + e^(-X))
//<2> Y' = Y - Y^2 = Y * (1 - Y)

__global__ void sigmoid2D_deltaX_v2_kernel_4(
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

		float4 y;//<1> Y = 1 / (1 + e^(-X))
		y.x = 1.0f / (1.0f + expf(-x.x));
		y.y = 1.0f / (1.0f + expf(-x.y));
		y.z = 1.0f / (1.0f + expf(-x.z));
		y.w = 1.0f / (1.0f + expf(-x.w));

		float4 dx;//<2> Y' = Y * (1 - Y)
		dx.x = y.x * (1.0f - y.x);
		dx.y = y.y * (1.0f - y.y);
		dx.z = y.z * (1.0f - y.z);
		dx.w = y.w * (1.0f - y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __sigmoid2D_deltaX_v2(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* X,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sigmoid2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	sigmoid2d_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, lengthv, width, stride);
}

#endif