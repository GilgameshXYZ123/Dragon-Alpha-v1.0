#pragma once

#ifndef SOFTPLUS_2D_H
#define SOFTPLUS_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SOFTPLUS_2D_CALL
#define SOFTPLUS_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define softplus2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	softplus2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define softplus2d_k4_small(stream, X, Y, lengthv, width, stride)\
	softplus2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef SOFTPLUS_2D_KERNEL
#define SOFTPLUS_2D_KERNEL

__global__ void softplus2D_kernel_4(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y;//y = softPlus(x) = log(1 + e^x) = log1p(exp(x)).
		y.x = log1pf(expf(x.x));
		y.y = log1pf(expf(x.y));
		y.z = log1pf(expf(x.z));
		y.w = log1pf(expf(x.w));

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __softplus2D(cudaStream_t stream,
	const float* X,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { softplus2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	softplus2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif