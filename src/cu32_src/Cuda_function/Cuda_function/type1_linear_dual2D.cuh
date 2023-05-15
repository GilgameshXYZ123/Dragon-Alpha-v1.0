#pragma once

#ifndef LINEAR_DUAL_2D_H
#define LINEAR_DUAL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LINEAR_DUAL_2D_CALL
#define LINEAR_DUAL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define linear_dual2d_k4_small(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)

#define linear_dual2d_k4(stream, LB, LT, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)\
	linear_dual2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, alpha, beta, gamma, Y, lengthv, width, stride)

#endif


#ifndef LINEAR_DUAL_2D_KERNEL
#define LINEAR_DUAL_2D_KERNEL

//Y = alpha*X1 + beta*X2 + gamma,
//deltaX1 = deltaY*alpha
//deltaX2 = deltaY*beta
__global__ void linear_dual2D_kernel_4(
	const float* __restrict__ X1, 
	const float* __restrict__ X2,
	float alpha, float beta, float gamma,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float4 x2 = *(float4*)(X2 + index4);

		float4 y; 
		y.x = alpha * x1.x + beta * x2.x + gamma;
		y.y = alpha * x1.y + beta * x2.y + gamma;
		y.z = alpha * x1.z + beta * x2.z + gamma;
		y.w = alpha * x1.w + beta * x2.w + gamma;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __linear_dual2D(cudaStream_t stream,
	const float* X1, 
	const float* X2, 
	float alpha, float beta, float gamma,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { linear_dual2d_k4_small(stream, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride); return; }
	linear_dual2d_k4(stream, 5, 2, X1, X2, alpha, beta, gamma, Y, lengthv, width, stride);
}

#endif