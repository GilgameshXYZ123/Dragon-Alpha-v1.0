#pragma once

#ifndef MIN_2D_H
#define MIN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef MIN_2D_CALL
#define MIN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define min2d_k4(stream, LB, LT, alpha, X, beta, vmin, Y, lengthv, width, stride)\
	min2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, vmin, Y, lengthv, width, stride)

#define min2d_k4_small(stream, alpha, X, beta, vmin, Y, lengthv, width, stride)\
	min2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, vmin, Y, lengthv, width, stride)

#endif


#ifndef MIN_2D_KERNEL
#define MIN_2D_KERNEL

__global__ void min2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta, 
	float vmin,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; simdLinear4(y, alpha, x, beta);
		y.x = fminf(y.x, vmin);
		y.y = fminf(y.y, vmin);
		y.z = fminf(y.z, vmin);
		y.w = fminf(y.w, vmin);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __min2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float vmin,
	float* Y, 
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { min2d_k4_small(stream, alpha, X, beta, vmin, Y, lengthv, width, stride); return; }
	min2d_k4(stream, 5, 2, alpha, X, beta, vmin, Y, lengthv, width, stride);
}

#endif