#pragma once

#ifndef SIGMOID_2D_H
#define SIGMOID_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SIGMOID_2D_CALL
#define SIGMOID_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sigmoid2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	sigmoid2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define sigmoid2d_k4_small(stream, X, Y, lengthv, width, stride)\
	sigmoid2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef SIGMOID_2D_KERNEL
#define SIGMOID_2D_KERNEL

__global__ void sigmoid2D_kernel_4(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		
		float4 y;//sigmoid(x) = 1/(1 + exp(-x))
		y.x = 1.0f / (1.0f + expf(-x.x));
		y.y = 1.0f / (1.0f + expf(-x.y));
		y.z = 1.0f / (1.0f + expf(-x.z));
		y.w = 1.0f / (1.0f + expf(-x.w));

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __sigmoid2D(cudaStream_t stream,
	const float* X,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sigmoid2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	sigmoid2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif