#pragma once

#ifndef SIGMOID_2D_DELTAX_V1_H
#define SIGMOID_2D_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
#ifndef SIGMOID_2D_DELTAX_V1_CALL
#define SIGMOID_2D_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sigmoid2d_deltaX_v1_k4(stream, LB, LT, deltaX, deltaY, Y, lengthv, width, stride)\
	sigmoid2D_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

#define sigmoid2d_deltaX_v1_k4_small(stream, deltaX, deltaY, Y, lengthv, width, stride)\
	sigmoid2D_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

#endif


#ifndef SIGMOID_2D_DELTAX_V1_KERNEL
#define SIGMOID_2D_DELTAX_V1_KERNEL

//Y = 1/(1 + e^(-X))
//Y' = Y - Y^2 = Y(1 - Y)
#define SIGMOID_DERI(y) (y * (1.0f - y))

__global__ void sigmoid2D_deltaX_v1_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 y = *(float4*)(Y + index4);

		float4 dx;//find derivative
		dx.x = SIGMOID_DERI(y.x);
		dx.y = SIGMOID_DERI(y.y);
		dx.z = SIGMOID_DERI(y.z);
		dx.w = SIGMOID_DERI(y.w);
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __sigmoid2D_deltaX_v1(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sigmoid2d_deltaX_v1_k4_small(stream, deltaX, deltaY, Y, lengthv, width, stride); return; }
	sigmoid2d_deltaX_v1_k4(stream, 5, 2, deltaX, deltaY, Y, lengthv, width, stride);
}

#endif