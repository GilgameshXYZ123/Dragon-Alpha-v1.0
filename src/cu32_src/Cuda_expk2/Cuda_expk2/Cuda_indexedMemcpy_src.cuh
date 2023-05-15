#pragma once

#ifndef SRC_INDEXED_MEMCPY_H
#define SRC_INDEXED_MEMCPY_H

//lengthv = Y.lengthv 
//X.lengthv = Index.lengthv
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SRC_INDEXED_MEMCPY_CALL
#define SRC_INDEXED_MEMCPY_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define srcIndexedMemcpy_k4(stream, LB, LT, X, Index, Y, lengthv, stride, width)\
	src_indexed_memcpy_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Index, Y, lengthv, stride, width)

#define srcIndexedMemcpy_k4_small(stream, X, Index, Y, lengthv, stride, width)\
	src_indexed_memcpy_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Index, Y, lengthv, stride, width)

#endif


#ifndef SRC_INDEXED_MEMCPY_KERNEL
#define SRC_INDEXED_MEMCPY_KERNEL

__global__ void src_indexed_memcpy_kernel_4(
	const float* __restrict__ X,
	const int* __restrict__ Index,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int4 idx = *(int4*)(Index + index4);

		float4 y;// for i = 0:lengthv: Y[i] = X[Index[i]].
		y.x = X[idx.x];
		y.y = X[idx.y];
		y.z = X[idx.z];
		y.w = X[idx.w];

		//when X[idx] is not element of X, but is memAligned zero, idx = 0
		//but X[0] may be not = 0, it will harm the memAligned zero of Y
		within_width(y, index4, stride, width);

		*(float4*)(Y + index4) = y;
	}
}

#endif


void __srcIndexedMemcpy(cudaStream_t stream,
	const float* X, const int* Index,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { srcIndexedMemcpy_k4_small(stream, X, Index, Y, lengthv, stride, width); return; }
	srcIndexedMemcpy_k4(stream, 5, 2, X, Index, Y, lengthv, stride, width);
}

#endif