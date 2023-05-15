#pragma once

#ifndef TRANSPOSED_3D_H
#define TRANSPOSED_3D_H

//mul(Xdim) = length
//mul(Ydim) = length
#ifndef TRANSPOSED_3D_CALL
#define TRANSPOSED_3D_CALL

//LB = log2(BLOCK_SIZE)

#define tp3d_k1(stream, LB, LT, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)\
	transpose3D_kernel_1\
		<<< (length >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)

#define tp3d_k1_small(stream, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)\
	transpose3D_kernel_1\
		<<< 1, (length + 3) >> 2, 0, stream >>>\
			(X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length)

#define tp3d_k4(stream, LB, LT, X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, length)\
	transpose3D_kernel_4\
		<<< (lengthv >> LB >> LT), (1<<LB), 0, stream>>>\
			(X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, length)

#define tp3d_k4_small(stream, X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, lengthv)\
	transpose3D_kernel_4\
		<<< 1, (lengthv + 3) >> 2, 0, stream >>>\
			(X, Y, Xd1, Yd1, dIdx1, dIdx2, stride, lengthv)

#endif


#ifndef TRANSPOSE_3D_KERNEL_1
#define TRANSPOSE_3D_KERNEL_1

//Y = X^T
//(2, 0): Size = 32, Time = 0.936 mesc, Speed = 33.3868GB/s
//(3, 0): Size = 32, Time = 0.938 mesc, Speed = 33.3156GB/s
//(1, 2): Size = 32, Time = 0.936 mesc, Speed = 33.3868GB/s

//if dimIndex1 > dimIndex2: swap(dimIndex1, dimIndex2)
//so: dimIndex1 < dimIndex2
__global__ void transpose3D_kernel_1(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int Xdim1, int Xdim2,//Xdim0
	int Ydim1, int Ydim2,//Ydim0
	int dimIndex1, int dimIndex2,
	int strideX, int strideY, int length)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x);

	int Xdim12 = Xdim1 * Xdim2;

	int x[3];
	for (; index < length; index += step)
	{
		int xoffset = index;
		x[0] = xoffset / Xdim12; int xoffset_res = xoffset - x[0] * Xdim12;
		x[1] = xoffset_res / Xdim2; 
		x[2] = xoffset_res - x[1] * Xdim2;

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = (x[0] * Ydim1 + x[1])*Ydim2 + x[2];

		//consider the mem alignment
		xoffset = (xoffset / Xdim2)*strideX + (xoffset % Xdim2);
		yoffset = (yoffset / Ydim2)*strideY + (yoffset % Ydim2);

		Y[yoffset] = X[xoffset];
	}
}

#endif


//dimIndex2 < 2
//lengthv = length / Xdim2 * stride
#ifndef TRANSPOSE_3D_KERNEL_4
#define TRANSPOSE_3D_KERNEL_4

//if dimIndex2 < 2: 
//the tranpose is performed on the first three dim
//so the basic mem struture is not changed, and: Ydim2 = Xdim2
//we can use float4
//(1, 2): Size = 32, Time = 0.348 mesc, Speed = 89.7989GB/s
//(0, 2): Size = 32, Time = 0.352 mesc, Speed = 88.7784GB/s
//(0, 1): Size = 32, Time = 0.348 mesc, Speed = 89.7989GB/s
__global__ void transpose3D_kernel_4(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int Xdim1,//Xdim2 = strideX = stride(consider memory alignment)
	int Ydim1,//Ydim2 = strideY = stride
	int dimIndex1, int dimIndex2,
	int stride, int lengthv)
{
	int index = (blockIdx.x*blockDim.x) + threadIdx.x;
	int step = (blockDim.x * gridDim.x), step4 = step << 2;

	int Xdim12 = Xdim1 * stride;//Xdim2 = strideX = stride

	int x[3];
	for (int index4 = (index << 2); index4 < lengthv; index4 += step4)
	{
		int xoffset = index4;
		x[0] = xoffset / Xdim12; int xoffset_res = xoffset - x[0] * Xdim12;
		x[1] = xoffset_res / stride;//Xdim2 = stride
		x[2] = xoffset_res - x[1] * stride;//Xdim2 = stride

		int t = x[dimIndex1]; x[dimIndex1] = x[dimIndex2]; x[dimIndex2] = t;
		int yoffset = (x[0] * Ydim1 + x[1])*stride + x[2];//Xdim2 = stride

		*(float4*)(&Y[yoffset]) = *(float4*)(&X[xoffset]);
	}
}

#endif


void __transpose3d(cudaStream_t stream,
	const float* __restrict__ X,
	float* __restrict__ Y,
	int Xd1, int Xd2,//Xdim0
	int Yd1, int Yd2,//Ydim0
	int dIdx1, int dIdx2,
	int strideX, int strideY, //the mem_stride >= mem_width
	int length)
{
	//make sure: dIdx2 > dIdx1
	if (dIdx1 > dIdx2) { int t = dIdx1; dIdx1 = dIdx2; dIdx2 = t; }

	//dimIndex2: we must have, aligned_Xdim2 = aligned_Ydim2 = strideX = strideY = stride
	if (dIdx2 < 2) {//no change to basic mem structure
		int lengthv = length / Xd2 * strideX;// length / mem_width * mem_stride
		if (length < 256) { tp3d_k4_small(stream, X, Y, Xd1, Yd1, dIdx1, dIdx2, strideX, lengthv); return; }
		tp3d_k4(stream, 5, 2, X, Y, Xd1, Yd1, dIdx1, dIdx2, strideX, lengthv); return;
	}

	if ((dIdx2 == 2) && (dIdx1 == 1) && (Xd2 == strideX) && (Yd2 == strideY)) {
		int Batch = length / (Xd1 * Xd2);//Batch = Xd0
		int N = Xd1;
		int M = Xd2;
		if (__batch_mat_transpose(stream, X, Y, Batch, N, M)) return;
	}

	if (length < 256) { tp3d_k1_small(stream, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length); return; }
	tp3d_k1(stream, 5, 2, X, Y, Xd1, Xd2, Yd1, Yd2, dIdx1, dIdx2, strideX, strideY, length);
}

#endif