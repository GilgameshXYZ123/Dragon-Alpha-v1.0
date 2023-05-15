#pragma once

#ifndef LEAKY_RELU_2D_DELTAX_V1_H
#define LEAKY_RELU_2D_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
#ifndef LEAKY_RELU_2D_DELTAX_V1_CALL
#define LEAKY_RELU_2D_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define leakyRelu2d_deltaX_v1_k4(stream, LB, LT, deltaX, deltaY, Y, k, lengthv, width, stride)\
	leakyRelu2D_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, k, lengthv, width, stride)

#define leakyRelu2d_deltaX_v1_k4_small(stream,  deltaX, deltaY, Y, k, lengthv, width, stride)\
	leakyRelu2D_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, k, lengthv, width, stride)

#endif


#ifndef LEAKY_RELU_2D_DELTAX_V1_KERNEL
#define LEAKY_RELU_2D_DELTAX_V1_KERNEL

//<1> X >  0: Y = X    , Y  > 0
//<2> X <= 0: Y = k * X, Y <= 0
//(X >  0) -> (Y >  0)
//(X <= 0) -> (Y <= 0)
//(1) Y' = (Y > 0) + (Y <= 0)*k = 1 + (Y <= 0)*(k - 1)
//(2) let: k = k - 1
//(3) Y' = 1 + (Y <= 0)*k

#define LEAKY_RELU_DERI(y, k) (1.0f + (y<=0)*k)

__global__ void leakyRelu2D_deltaX_v1_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y, float k,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	k = k - 1.0f;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);

		float4 dx;//find derivative
		dx.x = LEAKY_RELU_DERI(y.x, k) * dy.x;
		dx.y = LEAKY_RELU_DERI(y.y, k) * dy.y;
		dx.z = LEAKY_RELU_DERI(y.z, k) * dy.z;
		dx.w = LEAKY_RELU_DERI(y.w, k) * dy.w;
		
		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __leakyRelu2D_deltaX_v1(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *Y, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { leakyRelu2d_deltaX_v1_k4_small(stream, deltaX, deltaY, Y, k, lengthv, width, stride); return; }
	leakyRelu2d_deltaX_v1_k4(stream, 5, 2, deltaX, deltaY, Y, k, lengthv, width, stride);
}

#endif