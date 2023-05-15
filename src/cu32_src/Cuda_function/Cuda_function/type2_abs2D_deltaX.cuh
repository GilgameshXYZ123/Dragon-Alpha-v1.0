#pragma once

#ifndef ABS_2D_DELTAX_H
#define ABS_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ABS_2D_DELTAX_CALL
#define ABS_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define abs2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	abs2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

#define abs2d_deltaX_k4_small(stream,  deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	abs2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

#endif


#ifndef ABS_2D_DELTAX_KERNEL
#define ABS_2D_DELTAX_KERNEL

//Y = abs(alpha * X + beta) = |alpha * X + beta|
//Y'= alpha * sign(alpha * X + beta)
__global__ void abs2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	float alpha, float beta,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4); 
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx;//Y' = alpha * sign(alpha*X + beta)
		x.x = alpha * x.x + beta; dx.x = alpha * SIGN(x.x);
		x.y = alpha * x.y + beta; dx.y = alpha * SIGN(x.y);
		x.z = alpha * x.z + beta; dx.z = alpha * SIGN(x.z);
		x.w = alpha * x.w + beta; dx.w = alpha * SIGN(x.w);
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __abs2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *X, 
	float alpha, float beta,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { abs2d_deltaX_k4_small(stream, deltaX, deltaY, X, alpha, beta, lengthv, width, stride); return; }
	abs2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, X, alpha, beta, lengthv, width, stride);
}

#endif