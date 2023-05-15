#pragma once 

#ifndef CUDA_FLOAT_SET_1TO2
#define CUDA_FLOAT_SET_1TO2

//XL = X.length
//X -> Y
//YS > YW
#ifndef CUDA_FLOAT_SET_1TO2_CALL
#define CUDA_FLOAT_SET_1TO2_CALL

#define set1to2_k4(stream, LB, LT, X, XL, Y, YW, YS) \
	set1to2_kernel_4\
		<<< (XL>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, XL, Y, YW, YS)

#define set1to2_k1(stream, X, XL, Y, YW, YS) \
	set1to2_kernel_1\
		<<< 1, XL, 0, stream >>>\
			(X, XL, Y, YW, YS)

#endif


#ifndef CUDA_FLOAT_SET_1TO2_KERNEL_4
#define CUDA_FLOAT_SET_1TO2_KERNEL_4

__global__ void set1to2_kernel_4(
	const float* __restrict__ X, int XL,
	      float* __restrict__ Y, int YW, int YS)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int Xindex = blockIdx.x*blockDim.x + threadIdx.x, Xindex4 = Xindex << 2;

	//use float4---------------------------------------------
	int I = XL / step4; 
	for (int i = 0; i < I; i++)
	{
		float4 x = *(float4*)(X+ Xindex4);
		int4 k;
		k.x = Xindex4;
		k.y = Xindex4 + 1;
		k.z = Xindex4 + 2;
		k.w = Xindex4 + 3;

		Y[k.x / YW * YS + k.x % YW] = x.x;
		Y[k.y / YW * YS + k.y % YW] = x.y;
		Y[k.z / YW * YS + k.z % YW] = x.z;
		Y[k.w / YW * YS + k.w % YW] = x.w;
		Xindex4 += step4;
	}
	//use float1---------------------------------------------
	for (Xindex += I * step4; Xindex < XL; Xindex += step)
		Y[Xindex / YW * YS + Xindex % YW] = X[Xindex];
}

#endif


#ifndef CUDA_FLOAT_SET_1TO2_KERNEL_1
#define CUDA_FLOAT_SET_1TO2_KERNEL_1

__global__ void set1to2_kernel_1(
	const float* __restrict__ X, int XL,
	      float* __restrict__ Y, int YW, int YS)
{
	int Xindex = threadIdx.x;
	int step = blockDim.x;
	while (Xindex < XL) 
	{
		Y[Xindex / YW * YS + Xindex % YW] = X[Xindex];
		Xindex += step;
	}
}

#endif


void __set1to2(cudaStream_t stream,
	float *X, int XL,
	float *Y, int YH, int YW, int YS)
{
	if (XL < 256) { set1to2_k1(stream, X, XL, Y, YW, YS); return; }
	set1to2_k4(stream, 5, 2, X, XL, Y, YW, YS);
}

#endif 
