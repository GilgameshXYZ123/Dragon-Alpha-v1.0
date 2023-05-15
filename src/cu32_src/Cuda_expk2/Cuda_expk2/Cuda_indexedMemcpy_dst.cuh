#pragma once

#ifndef DST_INDEXED_MEMCPY_H
#define DST_INDEXED_MEMCPY_H

//lengthv = Y.lengthv = Index.lengthv
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef DST_INDEXED_MEMCPY_CALL
#define DST_INDEXED_MEMCPY_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define dstIndexedMemcpy_k4(stream, LB, LT, X, Index, Y, lengthv, width, stride)\
	dst_indexed_memcpy_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Index, Y, lengthv, width, stride)

#define dstIndexedMemcpy_k4_small(stream, X, Index, Y, lengthv, width, stride)\
	dst_indexed_memcpy_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Index, Y, lengthv, width, stride)

#endif


#ifndef DST_INDEXED_MEMCPY_KERNEL
#define DST_INDEXED_MEMCPY_KERNEL

__global__ void dst_indexed_memcpy_kernel_4(
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
		float4 x = *(float4*)(X + index4);

		//when X[idx] is not element of X, but is memAligned zero, idx = 0
		//but X[0] may be not = 0, it will harm the memAligned zero of Y
		//for i = 0:lengthv: Y[Index[i]] = X[i].
		if(((index4    ) % stride) < width) Y[idx.x] = x.x;
		if(((index4 + 1) % stride) < width) Y[idx.y] = x.y;
		if(((index4 + 2) % stride) < width) Y[idx.z] = x.z;
		if(((index4 + 3) % stride) < width) Y[idx.w] = x.w;
	}
}

#endif


void __dstIndexedMemcpy(cudaStream_t stream,
	const float* X, const int* Index,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { dstIndexedMemcpy_k4_small(stream, X, Index, Y, lengthv, width, stride); return; }
	dstIndexedMemcpy_k4(stream, 5, 2, X, Index, Y, lengthv, width, stride);
}

#endif