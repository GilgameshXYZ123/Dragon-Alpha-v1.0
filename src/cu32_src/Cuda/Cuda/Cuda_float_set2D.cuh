#pragma once

#ifndef FLOAT_SET_2D_H
#define FLOAT_SET_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef FLOAT_SET_2D_CALL
#define FLOAT_SET_2D_CALL

#define set2d_k4(stream, LB, LT, X, value, lengthv, width, stride) \
	set_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, value, lengthv, width, stride)

#define set2d_k4_small(stream, X, value, lengthv, width, stride) \
	set_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, value, lengthv, width, stride)

#endif


#ifndef FLOAT_SET_2D_KERNEL_4
#define FLOAT_SET_2D_KERNEL_4

//lengthv = height * stride
__global__ void set_2D_kernel_4(
	float* __restrict__ X, 
	const float value,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4) 
	{
		float4 v = make_float4(value, value, value, value);

		within_width(v, index4, stride, width);
		*(float4*)(X + index4) = v;
	}
}

#endif


void __set2D(cudaStream_t stream,
	float* X,
	const float value,
	int height, int width, int stride)
{
	int lengthv = height * stride;
	if (lengthv < 256) { set2d_k4_small(stream, X, value, lengthv, width, stride); return; }
	set2d_k4(stream, 5, 2, X, value, lengthv, width, stride);
}

#endif