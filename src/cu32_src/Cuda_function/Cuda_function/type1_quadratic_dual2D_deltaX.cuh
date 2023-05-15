#pragma once

#ifndef QUADRATIC_DUAL_2D_DELTAX_H
#define QUADRATIC_DUAL_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef QUADRATIC_DUAL_2D_DELTAX_CALL
#define QUADRATIC_DUAL_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define quadratic_dual2d_deltaX_k4(stream, LB, LT, deltaX1, deltaX2, deltaY, X1, X2, k11, k12, k22, k1, k2, lengthv, width, stride)\
	quadratic_dual2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, k11, k12, k22, k1, k2, lengthv, width, stride)

#define quadratic_dual2d_deltaX_k4_small(stream, deltaX1, deltaX2, deltaY, X1, X2, k11, k12, k22, k1, k2, lengthv, width, stride)\
	quadratic_dual2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, X2, k11, k12, k22, k1, k2, lengthv, width, stride)

#endif


#ifndef QUADRATIC_DUAL_2D_DELTAX_KERNEL
#define QUADRATIC_DUAL_2D_DELTAX_KERNEL

//Y = k11*X1^2 + k12*X1*X2  + k22*X2^2 + k1*X1 + k2*X2 + C
//dY / dX1 = k11*2*X1 + k12*X2 + k1
//dY / dX2 = k22*2*X2 + k12*X1 + k2
//deltaX1 = (dY / dX1) * deltaY
//deltaX2 = (dY / dX2) * deltaY

__global__ void quadratic_dual2D_deltaX_kernel_4(
	float* __restrict__ deltaX1,
	float* __restrict__ deltaX2,
	const float* __restrict__ deltaY,
	const float* __restrict__ X1,
	const float* __restrict__ X2,
	float k11, float k12, float k22,
	float k1, float k2,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	k11 *= 2.0f; k22 *= 2.0f;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1     + index4);
		float4 x2 = *(float4*)(X2     + index4);
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx1;//dY / dX1 = k11*2*X1 + k12*X2 + k1
		dx1.x = (k11 * x1.x) + (k12 * x2.x) + k1;
		dx1.y = (k11 * x1.y) + (k12 * x2.y) + k1;
		dx1.z = (k11 * x1.z) + (k12 * x2.z) + k1;
		dx1.w = (k11 * x1.w) + (k12 * x2.w) + k1;

		float4 dx2;//dY / dX2 = k22*2*X2 + k12*X1 + k2
		dx2.x = (k22 * x2.x) + (k12 * x1.x) + k2;
		dx2.y = (k22 * x2.y) + (k12 * x1.y) + k2;
		dx2.z = (k22 * x2.z) + (k12 * x1.z) + k2;
		dx2.w = (k22 * x2.w) + (k12 * x1.w) + k2;
		
		simdMul4(dx1, dx1, dy);//deltaX1 = (dY / dX1) * deltaY
		simdMul4(dx2, dx2, dy);//deltaX2 = (dY / dX2) * deltaY
		
		within_width(dx1, index4, stride, width);
		within_width(dx2, index4, stride, width);
		*(float4*)(deltaX1 + index4) = dx1;
		*(float4*)(deltaX2 + index4) = dx2;
	}
}

#endif


void __quadratic_dual2D_deltaX(cudaStream_t stream,
	float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float *X1, const float* X2,
	float k11, float k12, float k22,
	float k1, float k2,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { quadratic_dual2d_deltaX_k4_small(stream, deltaX1, deltaX2, deltaY, X1, X2, k11, k12, k22, k1, k2, lengthv, width, stride); return; }
	quadratic_dual2d_deltaX_k4(stream, 5, 2, deltaX1, deltaX2, deltaY, X1, X2, k11, k12, k22, k1, k2, lengthv, width, stride);
}

#endif