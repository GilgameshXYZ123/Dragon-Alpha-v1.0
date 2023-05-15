#pragma once

#ifndef CLIP_2D_H
#define CLIP_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CLIP_2D_CALL
#define CLIP_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define clip2d_k4(stream, LB, LT, alpha, X, beta, vmin, vmax, Y, lengthv, width, stride)\
	clip2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(alpha, X, beta, vmin, vmax, Y, lengthv, width, stride)

#define clip2d_k4_small(stream, alpha, X, beta, vmin, vmax, Y, lengthv, width, stride)\
	clip2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(alpha, X, beta, vmin, vmax, Y, lengthv, width, stride)

#endif


#ifndef CLIP_2D_KERNEL
#define CLIP_2D_KERNEL

#define CLIP(x, vmin, vmax) ((x<=vmin)*vmin + (x<vmax && x>vmin)*x + (x>=vmax)*vmax)

__global__ void clip2D_kernel_4(
	float alpha, const float* __restrict__ X, float beta,
	float vmin, float vmax,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(&X[index4]);

		float4 y; simdLinear4(y, alpha, x, beta);
		y.x = CLIP(y.x, vmin, vmax);
		y.y = CLIP(y.y, vmin, vmax);
		y.z = CLIP(y.z, vmin, vmax);
		y.w = CLIP(y.w, vmin, vmax);

		within_width(y, index4, stride, width);
		*(float4*)(&Y[index4]) = y;
	}
}

#endif


void __clip2D(cudaStream_t stream,
	float alpha, const float* X, float beta,
	float vmin, float vmax,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { clip2d_k4_small(stream, alpha, X, beta, vmin, vmax, Y, lengthv, width, stride); return; }
	clip2d_k4(stream, 5, 2, alpha, X, beta, vmin, vmax, Y, lengthv, width, stride);
}

#endif