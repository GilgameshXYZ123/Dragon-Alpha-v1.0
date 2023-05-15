#pragma once

#ifndef L1_2D_H
#define L1_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef L1_2D_CALL
#define L1_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define L1_2d_k4(stream, LB, LT, Y, Yh, L, lengthv, width, stride)\
	L1_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#define L1_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride)\
	L1_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#endif


#ifndef L1_2D_KERNEL
#define L1_2D_KERNEL

__global__ void L1_2D_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float* __restrict__ L,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 loss;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y  = *(float4*)(Y  + index4);

		//L = |Yh - Y|
		loss.x = fabsf(yh.x - y.x);
		loss.y = fabsf(yh.y - y.y);
		loss.z = fabsf(yh.z - y.z);
		loss.w = fabsf(yh.w - y.w);

		within_width(loss, index4, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __L1_2D(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { L1_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride); return; }
	L1_2d_k4(stream, 5, 2, Y, Yh, L, lengthv, width, stride);
}

#endif