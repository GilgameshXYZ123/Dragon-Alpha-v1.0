#pragma once

#ifndef SIGN_2D_H
#define SIGN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SIGN_2D_CALL
#define SIGN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sign2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	sign2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#define sign2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	sign2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef SIGN_2D_KERNEL
#define SIGN_2D_KERNEL

#define SIGN(x) (((x)>0) - ((x)<0))

__global__ void sign2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(&X[index4]);

		float4 y; simdLinear4(y, alpha, x, beta);
		y.x = SIGN(y.x);
		y.y = SIGN(y.y);
		y.z = SIGN(y.z);
		y.w = SIGN(y.w);

		within_width(y, index4, stride, width);
		*(float4*)(&Y[index4]) = y;
	}
}

#endif


void __sign2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sign2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	sign2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif