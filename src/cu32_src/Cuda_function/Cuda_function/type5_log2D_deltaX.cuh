#pragma once

#ifndef LOG_2D_DELTAX_H
#define LOG_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LOG_2D_DELTAX_CALL
#define LOG_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define log2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	log2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#define log2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride)\
	log2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, lengthv, width, stride)

#endif


#ifndef LOG_2D_DELTAX_KERNEL
#define LOG_2D_DELTAX_KERNEL

//Y = log(alpha*X + beta);
//Y' = alpha / (alpha*X + beta) 
//Y = log(alpha*X + beta) -> exp(Y) = alpha*X + beta
//Y' = alpha * exp(-Y)
//deltaX = deltaY * Y' 
//deltaX = deltaY * alpha * exp(-Y)

__global__ void log2D_deltaX_kernel_4(
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

		float4 dx;//Y' = alpha * exp(-Y)
		dx.x = alpha * expf(-y.x);
		dx.y = alpha * expf(-y.y);
		dx.z = alpha * expf(-y.z);
		dx.w = alpha * expf(-y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __log2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* Y, float alpha,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { log2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, lengthv, width, stride); return; }
	log2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, alpha, lengthv, width, stride);
}

#endif