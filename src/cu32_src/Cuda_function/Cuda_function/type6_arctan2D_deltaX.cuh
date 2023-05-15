#pragma once

#ifndef ARCTAN_2D_DELTAX_H
#define ARCTAN_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ARCTAN_2D_DELTAX_CALL
#define ARCTAN_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define arctan2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	arctan2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#define arctan2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	arctan2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#endif


#ifndef ARCTAN_2D_DELTAX_KERNEL
#define ARCTAN_2D_DELTAX_KERNEL

//Y = arctan(alpha*X + beta)
//Y' = alpha / (1 + (alpha*X + beta)^2)
//As: alpha*X + beta = tan(Y)
//Y' = alpha / (1 + tan^(Y))
//Y' = alpha * cos^2(Y)

__global__ void arctan2D_deltaX_kernel_4(
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

		float4 dx;//Y' = alpha * cos^2(Y)
		y.x = cosf(y.x); dx.x = alpha * y.x * y.x;
		y.y = cosf(y.y); dx.y = alpha * y.y * y.y;
		y.z = cosf(y.z); dx.z = alpha * y.z * y.z;
		y.w = cosf(y.w); dx.w = alpha * y.w * y.w;
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __arctan2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* Y, float alpha,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { arctan2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride); return; }
	arctan2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, alpha, lengthv, width, stride);
}

#endif