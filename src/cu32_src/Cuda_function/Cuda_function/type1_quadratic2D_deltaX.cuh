#pragma once

#ifndef QUADRATIC_2D_DELTAX_H
#define QUADRATIC_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef QUADRATIC_2D_DELTAX_CALL
#define QUADRATIC_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define quadratic2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	quadratic2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

#define quadratic2d_deltaX_k4_small(stream,  deltaX, deltaY, X, alpha, beta, lengthv, width, stride)\
	quadratic2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, alpha, beta, lengthv, width, stride)

#endif


#ifndef QUADRATIC_2D_DELTAX_KERNEL
#define QUADRATIC_2D_DELTAX_KERNEL

//Quadratic: Y = alpha * X^2 + beta * X + gamma
//Y' = 2 * alpha * X + beta
__global__ void quadratic2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	float alpha, float beta,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	alpha *= 2.0f;//alpha -> 2*alpha
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x  = *(float4*)(X      + index4);
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx;//Y' = 2*alpha*X + beta
		dx.x = alpha * x.x + beta;
		dx.y = alpha * x.y + beta;
		dx.z = alpha * x.z + beta;
		dx.w = alpha * x.w + beta;
		
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __quadratic2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *X, float alpha, float beta,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { quadratic2d_deltaX_k4_small(stream, deltaX, deltaY, X, alpha, beta, lengthv, width, stride); return; }
	quadratic2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, X, alpha, beta, lengthv, width, stride);
}

#endif