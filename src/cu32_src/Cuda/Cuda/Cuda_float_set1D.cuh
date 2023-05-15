#pragma once

#ifndef FLOAT_SET_1D_H
#define FLOAT_SET_1D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef FLOAT_SET_1D_CALL
#define FLOAT_SET_1D_CALL

#define set1d_k4(stream, LB, LT, X, value, length) \
	set_1D_kernel_4\
		<<< (length>>LB>>LT), (1<<LB), 0, stream >>>\
			(X, value, length)

#define set1d_k1(stream, X, value, length)\
	set_1D_kernel_1\
		<<< 1, length, 0, stream >>>\
			(X, value, length)

#endif


#ifndef FLOAT_SET_1D_KERNEL_4
#define FLOAT_SET_1D_KERNEL_4

__global__ void set_1D_kernel_4(
	float* __restrict__ X,
	const float value,
	int length)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x, index4 = index << 2;

	//use float4---------------------------------------------
	int I = length / step4;
	float4 x = make_float4(value, value, value, value);
	for (int i = 0; i < I; i++) {
		*(float4*)(X + index4) = x;
		index4 += step4;
	}

	//use float----------------------------------------------
	for (index += I * step4; index < length; index += step)
		X[index] = value;
}

#endif


#ifndef SET_1D_KERNEL_1
#define SET_1D_KERNEL_1

__global__ void set_1D_kernel_1(
	float *X,
	const float value,
	int length)
{
	int index = threadIdx.x;
	int step = blockDim.x;
	while (index < length) {
		X[index] = value;
		index += step;
	}
}

#endif


void __set1D(cudaStream_t stream,
	float* __restrict__ X,
	const float value,
	int length)
{
	if (length < 256) { set1d_k1(stream, X, value, length); return; }
	set1d_k4(stream, 5, 2, X, value, length);
}

#endif
