#pragma once

#ifndef RPL_2D_DELTAX_H
#define RPL_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef RPL_2D_DELTAX_CALL
#define RPL_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define rpl2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, alpha, gamma, lengthv, width, stride)\
	rpl2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, gamma, lengthv, width, stride)

#define rpl2d_deltaX_k4_small(stream,  deltaX, deltaY, Y, alpha, gamma, lengthv, width, stride)\
	rpl2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, gamma, lengthv, width, stride)

#endif


#ifndef RPL_2D_DELTAX_KERNEL
#define RPL_2D_DELTAX_KERNEL

//Reciprocal: Y = alpha / (X + beta) + gamma
//So: 1/(X + beta) = (1 / alpha) * (Y - gamma)
//Y' = - alpha / (X + beta)^2
//Y' =  -(1 / alpha) * (Y - gamma)^2

__global__ void rpl2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float alpha, float gamma,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	alpha = -1.0f / alpha;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);
		
		float4 dx;//Y' =  -(1 / alpha) * (Y - gamma)^2
		y.x -= gamma; dx.x = alpha * (y.x * y.x);
		y.y -= gamma; dx.y = alpha * (y.y * y.y);
		y.z -= gamma; dx.z = alpha * (y.z * y.z);
		y.w -= gamma; dx.w = alpha * (y.w * y.w);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __rpl2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *Y, float alpha, float gamma,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { rpl2d_deltaX_k4_small(stream, deltaX, deltaY, Y, alpha, gamma, lengthv, width, stride); return; }
	rpl2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, alpha, gamma, lengthv, width, stride);
}

#endif