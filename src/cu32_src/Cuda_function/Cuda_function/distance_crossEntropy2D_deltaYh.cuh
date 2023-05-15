#pragma once

#ifndef CROSS_ENTROPY_2D_DELTAYH_H
#define CROSS_ENTROPY_2D_DELTAYH_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CROSS_ENTROPY_2D_DELTAYH_CALL
#define CROSS_ENTROPY_2D_DELTAYH_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define crossEntropy_2d_deltaYh_k4(stream, LB, LT, Y, Yh, deltaYh, lengthv, width, stride)\
	crossEntropy_2D_deltaYh_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, deltaYh, lengthv, width, stride)

#define crossEntropy_2d_deltaYh_k4_small(stream, Y, Yh, deltaYh, lengthv, width, stride)\
	crossEntropy_2D_deltaYh_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, deltaYh, lengthv, width, stride)

#endif


#ifndef CROSS_ENTROPY_2D_DELTAYH_KERNEL
#define CROSS_ENTROPY_2D_DELTAYH_KERNEL

//deltaYh = -Y/Yh 
__global__ void crossEntropy_2D_deltaYh_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float* __restrict__ deltaYh,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y  = *(float4*)(Y  + index4);

		float4 dyh;//deltaYh = -Y/Yh 
		dyh.x = -(y.x / yh.x);
		dyh.y = -(y.y / yh.y);
		dyh.z = -(y.z / yh.z); 
		dyh.w = -(y.w / yh.w);

		within_width_zero_nan(dyh, index4, table, stride, width);
		*(float4*)(deltaYh + index4) = dyh;
	}
}

#endif

void __crossEntropy_2D_deltaYh(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* deltaYh,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { crossEntropy_2d_deltaYh_k4_small(stream, Y, Yh, deltaYh, lengthv, width, stride); return; }
	crossEntropy_2d_deltaYh_k4(stream, 5, 2, Y, Yh, deltaYh, lengthv, width, stride);
}

#endif