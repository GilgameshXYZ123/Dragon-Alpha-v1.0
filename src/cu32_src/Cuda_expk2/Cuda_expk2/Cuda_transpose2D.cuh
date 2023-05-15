#pragma once

#ifndef TRANSPOSED_2D_H
#define TRANSPOSED_2D_H

//mul(Xdim) = length
//mul(Ydim) = length
//Obviously: dIdx1 = 0, dIdx2 = 1
#ifndef TRANSPOSED_2D_CALL
#define TRANSPOSED_2D_CALL

//LB = log2(BLOCK_SIZE)

#define tp2d_k1(stream, LB, LT, X, Y, Xd1, Yd1, strideX, strideY, length)\
	transpose2D_kernel_1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, Xd1, Yd1, strideX, strideY, length)

#define tp2d_k1_small(stream, X, Y, Xd1, Yd1, strideX, strideY, length)\
	transpose2D_kernel_1\
		<<< 1, (length + 3) >> 2, 0, stream >>>\
			(X, Y, Xd1, Yd1, strideX, strideY, length)

#endif


#ifndef TRANSPOSE_2D_KERNEL_1
#define TRANSPOSE_2D_KERNEL_1

//Y = X^T
//(2, 0): Size = 32, Time = 0.936 mesc, Speed = 33.3868GB/s
//(3, 0): Size = 32, Time = 0.938 mesc, Speed = 33.3156GB/s
//(1, 2): Size = 32, Time = 0.936 mesc, Speed = 33.3868GB/s
//if dimIndex1 > dimIndex2: swap(dimIndex1, dimIndex2)
//so: dimIndex1 < dimIndex2
__global__ void transpose2D_kernel_1(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int Xdim1,//Ydim1 = Xdim0
	int Ydim1,//Ydim0 = Xdim1
	int strideX, int strideY, int length)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	int x0, x1;
	for (; index < length; index += step)
	{
		int xoffset = index;
		x0 = xoffset / Xdim1;
		x1 = xoffset - x0 * Xdim1;

		//swap(x1, x0): x0*Ydim1 + x1 ->
		int yoffset = x1 * Ydim1 + x0;

		//consider the mem alignment
		xoffset = (xoffset / Xdim1)*strideX + (xoffset % Xdim1);
		yoffset = (yoffset / Ydim1)*strideY + (yoffset % Ydim1);

		Y[yoffset] = X[xoffset];
	}
}

#endif


void __transpose2d(cudaStream_t stream,
	const float* __restrict__ X,
	float* __restrict__ Y,
	int Xd1,//Xdim0 = Ydim1
	int Yd1,//Ydim0 = Xdim1
	int strideX, int strideY, //the mem_stride >= mem_width
	int length)
{
	int N = strideY;//include the padded 0, so Xdim0 = Ydim1 -> strideY
	int M = strideX;//include the padded 0, so Xdim1 -> strideX
	if (__mat_transpose(stream, X, Y, N, M)) return;

	if (length < 256) { tp2d_k1_small(stream, X, Y, Xd1, Yd1, strideX, strideY, length); return; }
	tp2d_k1(stream, 5, 2, X, Y, Xd1, Yd1, strideX, strideY, length);
}

#endif
