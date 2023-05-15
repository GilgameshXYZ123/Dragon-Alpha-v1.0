#pragma once

#ifndef SOFTPLUS_2D_DELTAX_V2_H
#define SOFTPLUS_2D_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef SOFTPLUS_2D_DELTAX_V2_CALL
#define SOFTPLUS_2D_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define softplus2d_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, lengthv, width, stride)\
	softplus2D_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#define softplus2d_deltaX_v2_k4_small(stream,  deltaX, deltaY, X, lengthv, width, stride)\
	softplus2D_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, lengthv, width, stride)

#endif


#ifndef SOFTPLUS_2D_DELTAX_V2_KERNEL
#define SOFTPLUS_2D_DELTAX_V2_KERNEL

//Y = softplus(X) = log(exp(X) + 1)
//Y' = e^X / (1 + e^X)
//Y' = 1 / (1 + exp(-X))

__global__ void softplus2D_deltaX_v2_kernel_4(
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

		float4 dx;//Y' = 1 / (1 + exp(-X))
		dx.x = 1.0f / (1.0f + expf(-x.x));
		dx.y = 1.0f / (1.0f + expf(-x.y));
		dx.z = 1.0f / (1.0f + expf(-x.z));
		dx.w = 1.0f / (1.0f + expf(-x.w));
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __softplus2D_deltaX_v2(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* X,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { softplus2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, lengthv, width, stride); return; }
	softplus2d_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, lengthv, width, stride);
}

#endif