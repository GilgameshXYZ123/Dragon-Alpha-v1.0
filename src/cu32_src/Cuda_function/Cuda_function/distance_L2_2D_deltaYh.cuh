#pragma once

#ifndef L2_2D_DELTAYH_H
#define L2_2D_DELTAYH_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef L2_2D_DELTAYH_CALL
#define L2_2D_DELTAYH_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define L2_2d_deltaYh_k4(stream, LB, LT, Y, Yh, deltaYh, lengthv, width, stride)\
	L2_2D_deltaYh_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, deltaYh, lengthv, width, stride)

#define L2_2d_deltaYh_k4_small(stream, Y, Yh, deltaYh, lengthv, width, stride)\
	L2_2D_deltaYh_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, deltaYh, lengthv, width, stride)

#endif


#ifndef L2_2D_DELTAYH_KERNEL
#define L2_2D_DELTAYH_KERNEL

__global__ void L2_2D_deltaYh_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float* __restrict__ deltaYh,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 dyh;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y  = *(float4*)(Y  + index4);

		//deltaYh = Yh - Y
		dyh.x = 2.0f * (yh.x - y.x);
		dyh.y = 2.0f * (yh.y - y.y);
		dyh.z = 2.0f * (yh.z - y.z);
		dyh.w = 2.0f * (yh.w - y.w);

		within_width(dyh, index4, stride, width);
		*(float4*)(deltaYh + index4) = dyh;
	}
}

#endif


void __L2_2D_deltaYh(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* deltaYh,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { L2_2d_deltaYh_k4_small(stream, Y, Yh, deltaYh, lengthv, width, stride); return; }
	L2_2d_deltaYh_k4(stream, 5, 2, Y, Yh, deltaYh, lengthv, width, stride);
}

#endif