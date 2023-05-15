#pragma once

#ifndef LEAKY_RELU2D_H
#define LEAKY_RELU2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LEAKY_RELU2D_CALL
#define LEAKY_RELU2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define leakyRelu2d_k4(stream, LB, LT, X, k, Y, lengthv, width, stride)\
	leakyRelu2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, k, Y, lengthv, width, stride)

#define leakyRelu2d_k4_small(stream, X, k, Y, lengthv, width, stride)\
	leakyRelu2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, k, Y, lengthv, width, stride)

#endif


#ifndef LEAKY_RELU2D_KERNEL
#define LEAKY_RELU2D_KERNEL


#define LEAKY_RELU(x, k) (x * ((x > 0) + k*(x < 0)))

//this follow expression may reduce accuarcy
//#define LEAKY_RELU(x, k) (x * ((x>0)*(1.0f-k) + (k)))

__global__ void leakyRelu2D_kernel_4(
	const float* __restrict__ X, float k,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		float4 y;
		y.x = LEAKY_RELU(x.x, k);
		y.y = LEAKY_RELU(x.y, k);
		y.z = LEAKY_RELU(x.z, k);
		y.w = LEAKY_RELU(x.w, k);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __leakyRelu2D(cudaStream_t stream,
	const float* X, float k,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { leakyRelu2d_k4_small(stream, X, k, Y, lengthv, width, stride); return; }
	leakyRelu2d_k4(stream, 5, 2, X, k, Y, lengthv, width, stride);
}

#endif