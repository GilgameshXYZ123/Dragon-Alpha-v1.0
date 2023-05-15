#pragma once

#ifndef RECIPROCAL_2D_H
#define RECIPROCAL_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef RECIPROCAL_2D_CALL
#define RECIPROCAL_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define rpl2d_k4(stream, LB, LT, alpha, X, beta, gamma, Y, lengthv, width, stride)\
	rpl2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, gamma, Y, lengthv, width, stride)

#define rpl2d_k4_small(stream, alpha, X, beta, gamma, Y, lengthv, width, stride)\
	rpl2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, gamma, Y, lengthv, width, stride)

#endif


#ifndef RECIPROCAL_2D_KERNEL
#define RECIPROCAL_2D_KERNEL

//Reciprocal: Y = alpha / (X + beta) + gamma
__global__ void rpl2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta, float gamma,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; //Y = alpha / (X + beta) + gamma
		y.x = alpha / (x.x + beta) + gamma;
		y.y = alpha / (x.y + beta) + gamma;
		y.z = alpha / (x.z + beta) + gamma;
		y.w = alpha / (x.w + beta) + gamma;

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __rpl2D(cudaStream_t stream,
	float alpha, const float* X, float beta, float gamma,
	float* Y,
	int lengthv, int width, int stride)
{
	MOVE_E(beta); 
	if (lengthv < 256) { rpl2d_k4_small(stream, alpha, X, beta, gamma, Y, lengthv, width, stride); return; }
	rpl2d_k4(stream, 5, 2, alpha, X, beta, gamma, Y, lengthv, width, stride);
}

#endif