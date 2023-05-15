#pragma once

#ifndef RELU_2D_H
#define RELU_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef RELU_2D_CALL
#define RELU_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define relu2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	relu2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define relu2d_k4_small(stream, X, Y, lengthv, width, stride)\
	relu2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef RELU_2D_KERNEL
#define RELU_2D_KERNEL

#define RELU(x) fmaxf(x, 0.0f)

__global__ void relu2D_kernel_4(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y;
		y.x = RELU(x.x);
		y.y = RELU(x.y);
		y.z = RELU(x.z);
		y.w = RELU(x.w);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __relu2D(cudaStream_t stream,
	const float* X, 
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { relu2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	relu2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif