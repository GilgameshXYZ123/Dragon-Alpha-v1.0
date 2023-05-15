#pragma once

#ifndef SPARSE_UNIFORM_2D_H
#define SPARSE_UNIFORM_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SPARSE_UNIFORM_2D_CALL
#define SPARSE_UNIFORM_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sparse_uniform2d_k4(stream, LB, LT, X, seed1, seed2, p, threshold, base, lengthv, width, stride)\
	sparse_uniform2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, seed1, seed2, p, threshold, base, lengthv, width, stride)

#define sparse_uniform2d_k4_small(stream, X, seed1, seed2, p, threshold, base, lengthv, width, stride)\
	sparse_uniform2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, seed1, seed2, p, threshold, base, lengthv, width, stride)

#endif


#ifndef SPARSE_UNIFORM_2D_KERNEL
#define SPARSE_UNIFORM_2D_KERNEL

__global__ void sparse_uniform2D_kernel_4(
	float* __restrict__ X,
	unsigned int seed1,
	unsigned int seed2,
	float p, float threshold, float base,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	seed1 = (seed1*THREAD_MUL +  index      ) & THREAD_MOD;
	seed2 = (seed2*THREAD_MUL + (index << 1)) & THREAD_MOD;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 r; simdNextFloat4(r, seed1);
		float4 b; simdNextFloat4(b, seed2);

		simdLinear4(r, threshold, r, base);
		r.x *= (b.x <= p);
		r.y *= (b.y <= p);
		r.z *= (b.z <= p);
		r.w *= (b.w <= p);

		within_width(r, index4, stride, width);
		*(float4*)(&X[index4]) = r;
	}
}

#endif


void __sparse_uniform2D(cudaStream_t stream,
	float* X,
	int seed1, int seed2,
	float p, float vmin, float vmax,
	int lengthv, int width, int stride)
{
	float base = vmin, threshold = vmax - vmin;
	if (lengthv < 256) { sparse_uniform2d_k4_small(stream, X, seed1, seed2, p, threshold, base, lengthv, width, stride); return; }
	sparse_uniform2d_k4(stream, 5, 2, X, seed1, seed2, p, threshold, base, lengthv, width, stride);
}

#endif