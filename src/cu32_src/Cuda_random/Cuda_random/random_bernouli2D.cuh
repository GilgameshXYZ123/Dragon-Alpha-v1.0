#pragma once

#ifndef BERNOULI_2D_H
#define BERNOULI_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef BERNOULI_2D_CALL
#define BERNOULI_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define bernouli2d_k4(stream, LB, LT, X, seed, p, v1, v2, lengthv, width, stride)\
	bernouli2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, seed, p, v1, v2, lengthv, width, stride)

#define bernouli2d_k4_small(stream, X, seed, p, v1, v2, lengthv, width, stride)\
	bernouli2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, seed, p, v1, v2, lengthv, width, stride)

#endif


#ifndef BERNOULI_2D_KERNEL
#define BERNOULI_2D_KERNEL

__global__ void bernouli2D_kernel_4(
	float* __restrict__ X,
	unsigned int seed,
	float p, float v1, float v2,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	seed = (seed*THREAD_MUL + index) & THREAD_MOD;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 r; simdNextFloat4(r, seed);
		r.x = BERNOULI(r.x, p, v1, v2);
		r.y = BERNOULI(r.y, p, v1, v2);
		r.z = BERNOULI(r.z, p, v1, v2);
		r.w = BERNOULI(r.w, p, v1, v2);

		within_width(r, index4, stride, width);
		*(float4*)(X + index4) = r;
	}
}

#endif


void __bernouli2D(cudaStream_t stream,
	float* X,
	int seed,
	float p, float v1, float v2,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { bernouli2d_k4_small(stream, X, seed, p, v1, v2, lengthv, width, stride); return; }
	bernouli2d_k4(stream, 5, 2, X, seed, p, v1, v2, lengthv, width, stride);
}

#endif