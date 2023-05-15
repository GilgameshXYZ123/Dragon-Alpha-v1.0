#pragma once

#ifndef MAX_DUAL_2D_H
#define MAX_DUAL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef MAX_DUAL_2D_CALL
#define MAX_DUAL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define max_dual2d_k4(stream, LB, LT, alpha1, X1, beta1, alpha2, X2, beta2, Y, lengthv, width, stride)\
	max_dual2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha1, X1, beta1, alpha2, X2, beta2, Y, lengthv, width, stride)

#define max_dual2d_k4_small(stream, alpha1, X1, beta1, alpha2, X2, beta2, Y, lengthv, width, stride)\
	max_dual2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha1, X1, beta1, alpha2, X2, beta2, Y, lengthv, width, stride)

#endif


#ifndef MAX_DUAL_2D_KERNEL
#define MAX_DUAL_2D_KERNEL

__global__ void max_dual2D_kernel_4(
	float alpha1, const float* __restrict__ X1, float beta1,
	float alpha2, const float* __restrict__ X2, float beta2,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		simdLinear4(x1, alpha1, x1, beta1);
		simdLinear4(x2, alpha2, x2, beta2);

		float4 y; 
		y.x = fmaxf(x1.x, x2.x);
		y.y = fmaxf(x1.y, x2.y);
		y.z = fmaxf(x1.z, x2.z);
		y.w = fmaxf(x1.w, x2.w);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __max_dual2D(cudaStream_t stream,
	float alpha1, const float* X1, float beta1,
	float alpha2, const float* X2, float beta2,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { max_dual2d_k4_small(stream, alpha1, X1, beta1, alpha2, X2, beta2, Y, lengthv, width, stride); return; }
	max_dual2d_k4(stream, 5, 2, alpha1, X1, beta1, alpha2, X2, beta2, Y, lengthv, width, stride);
}

#endif