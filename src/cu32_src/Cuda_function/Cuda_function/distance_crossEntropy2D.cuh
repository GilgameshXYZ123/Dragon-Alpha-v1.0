#pragma once

#ifndef CROSS_ENTROPY_2D_H
#define CROSS_ENTROPY_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef CROSS_ENTROPY_2D_CALL
#define CROSS_ENTROPY_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define crossEntropy_2d_k4(stream, LB, LT, Y, Yh, L, lengthv, width, stride)\
	crossEntropy_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#define crossEntropy_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride)\
	crossEntropy_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#endif


#ifndef CROSS_ENTROPY_2D_KERNEL
#define CROSS_ENTROPY_2D_KERNEL

//Multi-Classification:
//L = sum_row: -Y * log(Yh)

__global__ void crossEntropy_2D_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float* __restrict__ L,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y =  *(float4*)(Y  + index4);

		float4 loss;//L = -Y * log(Yh)
		loss.x = -y.x * logf(yh.x);
		loss.y = -y.y * logf(yh.y);
		loss.z = -y.z * logf(yh.z);
		loss.w = -y.w * logf(yh.w);

		within_width_zero_nan(loss, index4, table, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __crossEntropy_2D(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { crossEntropy_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride); return; }
	crossEntropy_2d_k4(stream, 5, 2, Y, Yh, L, lengthv, width, stride);
}

#endif