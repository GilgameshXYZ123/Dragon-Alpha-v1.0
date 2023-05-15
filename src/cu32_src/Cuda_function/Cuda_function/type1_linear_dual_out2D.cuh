#pragma once

#ifndef LINEAR_DUAL_OUT_2D_H
#define LINEAR_DUAL_OUT_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_DUAL_OUT_2D_CALL
#define LINEAR_DUAL_OUT_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define linear_dual_out2d_k4_small(stream, X, alpha1, beta1, alpha2, beta2, Y1, Y2, lengthv, width, stride)\
	linear_dual_out2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, alpha1, beta1, alpha2, beta2, Y1, Y2, lengthv, width, stride)

#define linear_dual_out2d_k4(stream, LB, LT, X, alpha1, beta1, alpha2, beta2, Y1, Y2, lengthv, width, stride)\
	linear_dual_out2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha1, beta1, alpha2, beta2, Y1, Y2, lengthv, width, stride)

#endif


#ifndef LINEAR_DUAL_OUT_2D_KERNEL
#define LINEAR_DUAL_OUT_2D_KERNEL

//Y1 = alpha1*X1 + beta1
//Y2 = alpha2*X2 + beta2
__global__ void linear_dual_out2D_kernel_4(
	const float* __restrict__ X,
	float alpha1, float beta1, 
	float alpha2, float beta2,
	float* __restrict__ Y1,
	float* __restrict__ Y2,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y1;//y1 = alpha1*X + beta1
		y1.x = alpha1 * x.x + beta1;
		y1.y = alpha1 * x.y + beta1;
		y1.z = alpha1 * x.z + beta1;
		y1.w = alpha1 * x.w + beta1;

		float4 y2;//y2 = alpha2*X + beta2
		y2.x = alpha2 * x.x + beta2;
		y2.y = alpha2 * x.y + beta2;
		y2.z = alpha2 * x.z + beta2;
		y2.w = alpha2 * x.w + beta2;

		within_width(y1, index4, stride, width);
		within_width(y2, index4, stride, width);
		*(float4*)(Y1 + index4) = y1;
		*(float4*)(Y2 + index4) = y2;
	}
}

#endif


void __linear_dual_out2D(cudaStream_t stream,
	const float* X,
	float alpha1, float beta1,
	float alpha2, float beta2,
	float* Y1, float* Y2,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_dual_out2d_k4_small(stream, X, alpha1, beta1, alpha2, beta2, Y1, Y2, lengthv, width, stride); return; }
	linear_dual_out2d_k4(stream, 5, 2, X, alpha1, beta1, alpha2, beta2, Y1, Y2, lengthv, width, stride);
}

#endif