#pragma once

#ifndef UNIFORM_2D_H
#define UNIFORM_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef UNIFORM_2D_CALL
#define UNIFORM_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define uniform2d_k4(stream, LB, LT, X, seed, threshold, base, lengthv, width, stride)\
	uniform2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, seed, threshold, base, lengthv, width, stride)

#define uniform2d_k4_small(stream, X, seed, threshold, base, lengthv, width, stride)\
	uniform2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, seed, threshold, base, lengthv, width, stride)

#endif


#ifndef UNIFORM_2D_KERNEL
#define UNIFORM_2D_KERNEL

__global__ void uniform2D_kernel_4(
	float* __restrict__ X, 
	unsigned int seed,
	float threshold, float base,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	seed = (seed*step + index) & THREAD_MOD;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x; simdNextFloat4(x, seed);
		simdLinear4(x, threshold, x, base);

		within_width(x, index4, stride, width);
		*(float4*)(X + index4) = x;
	}
}

#endif


void __uniform2D(cudaStream_t stream,
	float* X,	
	int seed, 
	float vmin, float vmax,
	int lengthv, int width, int stride)
{
	float base = vmin, threshold = vmax - vmin;
	if (lengthv < 256) { uniform2d_k4_small(stream, X, seed, threshold, base, lengthv, width, stride); return; }
	uniform2d_k4(stream, 5, 2, X, seed, threshold, base, lengthv, width, stride);
}

#endif