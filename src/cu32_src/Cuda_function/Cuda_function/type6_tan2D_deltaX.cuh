#pragma once

#ifndef TAN_2D_DELTAX_H
#define TAN_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef TAN_2D_DELTAX_CALL
#define TAN_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define tan2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	tan2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#define tan2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	tan2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#endif


#ifndef TAN_2D_DELTAX_KERNEL
#define TAN_2D_DELTAX_KERNEL

//Y = tan(alpha*X + beta) =
//Y' = alpha * (1 + tan^2(alpha*X + beta))
//Y' = alpha * (1 + y^2) = alpha * y^2 + alpha
//Y' = quadratic(alpha, 0, alpha)
//Y' = alpha * (y^2 + 1)
//deltaX = deltaY <*> (alpha * y^2 + alpha)
//deltaX = deltaY <*> alpha * (y^2 + 1)

__global__ void tan2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float alpha,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx;//Y' = alpha * (y^2 + 1)
		dx.x = alpha * (1.0f + y.x * y.x);
		dx.y = alpha * (1.0f + y.y * y.y);
		dx.z = alpha * (1.0f + y.z * y.z);
		dx.w = alpha * (1.0f + y.w * y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __tan2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* Y, float alpha,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { tan2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride); return; }
	tan2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, alpha, lengthv, width, stride);
}

#endif