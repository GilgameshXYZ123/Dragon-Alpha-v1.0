#pragma once

#ifndef SMOOTH_L1_2D_H
#define SMOOTH_L1_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SMOOTH_L1_2D_CALL
#define SMOOTH_L1_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define smoothL1_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride)\
	smoothL1_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#define smoothL1_2d_k4(stream, LB, LT, Y, Yh, L, lengthv, width, stride)\
	smoothL1_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#endif


#ifndef SMOOTH_L1_2D_KENREL
#define SMOOTH_L1_2D_KERNEL

__global__ void smoothL1_2D_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float* __restrict__ L,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y  = *(float4*)(Y  + index4);

		float4 loss;//L = (div - 0.5) + (1>=div)*(0.5*div*div + 0.5 -div), div = |Yh - Y|
		loss.x = fabsf(yh.x - y.x); loss.x = (loss.x - 0.5f) + (1 > loss.x)*(0.5f*loss.x*loss.x + 0.5f - loss.x);
		loss.y = fabsf(yh.y - y.y); loss.y = (loss.y - 0.5f) + (1 > loss.y)*(0.5f*loss.y*loss.y + 0.5f - loss.y);
		loss.z = fabsf(yh.z - y.z); loss.z = (loss.z - 0.5f) + (1 > loss.z)*(0.5f*loss.z*loss.z + 0.5f - loss.z);
		loss.w = fabsf(yh.w - y.w); loss.w = (loss.w - 0.5f) + (1 > loss.w)*(0.5f*loss.w*loss.w + 0.5f - loss.w);

		within_width(loss, index4, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __smoothL1_2D(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { smoothL1_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride); return; }
	smoothL1_2d_k4(stream, 5, 2, Y, Yh, L, lengthv, width, stride);
}

#endif