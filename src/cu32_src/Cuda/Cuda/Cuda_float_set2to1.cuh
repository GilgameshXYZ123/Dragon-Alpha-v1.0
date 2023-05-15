#pragma once 

#ifndef CUDA_FLOAT_SET_2TO1
#define CUDA_FLOAT_SET_2TO1

//YW < YS
//YH * YW = XL
#ifndef CUDA_FLOAT_SET_2TO1_CALL
#define CUDA_FLOAT_SET_2TO1_CALL

#define set2to1_k4(stream, LB, LT, X, XW, XS, Y, YL) \
	set2to1_kernel_4\
		<<< (YL>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, XW, XS, Y, YL)

#define set2to1_k1(stream, X, XW, XS, Y, YL) \
	set2to1_kernel_1\
		<<< 1, YL, 0, stream >>>\
			(X, XW, XS, Y, YL)

#endif


#ifndef CUDA_FLOAT_SET_2TO1_KERNEL_4
#define CUDA_FLOAT_SET_2TO1_KERNEL_4

__global__ void set2to1_kernel_4(
	const float* __restrict__ X, int XW, int XS,
	      float* __restrict__ Y, int YL)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int Yindex = blockIdx.x*blockDim.x + threadIdx.x, Yindex4 = Yindex << 2;

	//stage1: X.value -> Y.value-----------------------------
	//use float4---------------------------------------------
	int I = YL / step4;
	for (int i = 0; i < I; i++)
	{
		int4 k;
		k.x = Yindex4;
		k.y = Yindex4 + 1;
		k.z = Yindex4 + 2;
		k.w = Yindex4 + 3;

		float4 x;
		x.x = X[k.x / XW * XS + k.x % XW];
		x.y = X[k.y / XW * XS + k.y % XW];
		x.z = X[k.z / XW * XS + k.z % XW];
		x.w = X[k.w / XW * XS + k.w % XW];
		*(float4*)Y = x;
		Yindex += step4;
	}
	//use float1---------------------------------------------
	for (Yindex += I * step4; Yindex < YL; Yindex += step)
		Y[Yindex / XW * XS + Yindex % XW] = X[Yindex];
}

#endif


#ifndef CUDA_FLOAT_SET_2TO1_KERNEL_1
#define CUDA_FLOAT_SET_2TO1_KERNEL_1

__global__ void set2to1_kernel_1(
	const float* __restrict__ X, int XW, int XS,
	      float* __restrict__ Y, int YL)
{
	int Yindex = threadIdx.x;
	int step = blockDim.x;
	while (Yindex < YL) {
		Y[Yindex] = X[Yindex / XW * XS + Yindex % XW];
		Yindex += step;
	}
}

#endif


void __set2to1(cudaStream_t stream,
	float *X, int XH, int XW, int XS,
	float *Y, int YL)
{
	if (YL < 256) { set2to1_k1(stream, X, XW, XS, Y, YL); return; }
	set2to1_k4(stream, 5, 2, X, XW, XS, Y, YL);
}

#endif 
