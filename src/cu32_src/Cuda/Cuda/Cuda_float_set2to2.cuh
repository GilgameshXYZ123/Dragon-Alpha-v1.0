#pragma once 

#ifndef CUDA_FLOAT_SET_2TO2
#define CUDA_FLOAT_SET_2TO2

//XW < XS, YW < YS
//XW != YW
//XS != YS
//length = YH * YW = XH * XW
#ifndef CUDA_FLOAT_SET_2TO2_CALL
#define CUDA_FLOAT_SET_2TO2_CALL

#define set2to2_k1(stream, LB, LT, X, XW, XS, Y, YW, YS, length) \
	set2to2_kernel_1\
		<<< (length>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, XW, XS, Y, YW, YS, length)

#define set2to2_k1_simple(stream, X, XW, XS, Y, YW, YS, length) \
	set2to2_kernel_1\
		<<< 1, length, 0, stream >>>\
			(X, XW, XS, Y, YW, YS, length)

#endif


#ifndef CUDA_FLOAT_SET_2TO2_KERNEL_1
#define CUDA_FLOAT_SET_2TO2_KERNEL_1

//Y = X
__global__ void set2to2_kernel_1(
	const float* __restrict__ X, int XW, int XS,
	      float* __restrict__ Y, int YW, int YS,
	int length)
{
	int step = gridDim.x*blockDim.x;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	while (index < length) {
		int Xindex = (index / XW) * XS + (index % XW);
		int Yindex = (index / YW) * YS + (index % YW);

		Y[Yindex] = X[Xindex];
		index += step;
	}
}

#endif


void __set2to2(cudaStream_t stream,
	float *X, int XH, int XW, int XS,
	float *Y, int YH, int YW, int YS)
{
	int length = YH * YW;
	if (length < 256) { set2to2_k1_simple(stream, X, XW, XS, Y, YW, YS, length); return; }
	set2to2_k1(stream, 5, 2, X, XW, XS, Y, YW, YW, length);
}

#endif 
