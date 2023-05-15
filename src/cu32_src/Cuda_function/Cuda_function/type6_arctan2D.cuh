#pragma once

#ifndef ARCTAN_2D_H
#define ARCTAN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ARCTAN_2D_CALL
#define ARCTAN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define arctan2d_k4(stream, LB, LT, alpha, X, beta, Y, lengthv, width, stride)\
	arctan2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#define arctan2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride)\
	arctan2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef ARCTAN_2D_KERNEL
#define ARCTAN_2D_KERNEL

__global__ void arctan2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y; //Y = arctan(alpha*X + beta)
		y.x = atanf(alpha*x.x + beta);
		y.y = atanf(alpha*x.y + beta);
		y.z = atanf(alpha*x.z + beta);
		y.w = atanf(alpha*x.w + beta);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __arctan2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { arctan2d_k4_small(stream, alpha, X, beta, Y, lengthv, width, stride); return; }
	arctan2d_k4(stream, 5, 2, alpha, X, beta, Y, lengthv, width, stride);
}

#endif