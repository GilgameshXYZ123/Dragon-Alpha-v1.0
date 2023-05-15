#pragma once

#ifndef MAX_2D_H
#define MAX_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef MAX_2D_CALL
#define MAX_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define max2d_k4(stream, LB, LT, alpha, X, beta, vmax, Y, lengthv, width, stride)\
	max2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, vmax, Y, lengthv, width, stride)

#define max2d_k4_small(stream, alpha, X, beta, vmax, Y, lengthv, width, stride)\
	max2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, vmax, Y, lengthv, width, stride)

#endif


#ifndef MAX_2D_KERNEL
#define MAX_2D_KERNEL

__global__ void max2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float vmax,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; simdLinear4(y, alpha, x, beta);
		y.x = fmaxf(y.x, vmax);
		y.y = fmaxf(y.y, vmax);
		y.z = fmaxf(y.z, vmax);
		y.w = fmaxf(y.w, vmax);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __max2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float vmax,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { max2d_k4_small(stream, alpha, X, beta, vmax, Y, lengthv, width, stride); return; }
	max2d_k4(stream, 5, 2, alpha, X, beta, vmax, Y, lengthv, width, stride);
}

#endif