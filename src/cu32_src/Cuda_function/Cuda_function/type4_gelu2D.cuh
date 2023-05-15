#pragma once

#ifndef GELU_2D_H
#define GELU_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef GELU_2D_CALL
#define GELU_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define gelu2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	gelu2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define gelu2d_k4_small(stream, X, Y, lengthv, width, stride)\
	gelu2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


//GELU(x) = x * sigmoid(1.702x)
//= x * [1 / (1 + exp(-1.702x))]
//y = x/[1 + exp(-1.702x)]

__global__ void gelu2D_kernel_4(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(&X[index4]);

		float4 y;//y = x / [1 + exp(-1.702x)]
		y.x = x.x / (1.0f + expf(-1.702f * x.x));
		y.y = x.y / (1.0f + expf(-1.702f * x.y));
		y.z = x.z / (1.0f + expf(-1.702f * x.z));
		y.w = x.w / (1.0f + expf(-1.702f * x.w));

		within_width(y, index4, stride, width);
		*(float4*)(&Y[index4]) = y;
	}
}

void __gelu2D(cudaStream_t stream,
	const float* X,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { gelu2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	gelu2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif