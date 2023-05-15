#pragma once

#ifndef L2_2D_H
#define L2_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef L2_2D_CALL
#define L2_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define L2_2d_k4(stream, LB, LT, Y, Yh, L, lengthv, width, stride)\
	L2_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#define L2_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride)\
	L2_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, L, lengthv, width, stride)

#endif


#ifndef L2_2D_KERNEL
#define L2_2D_KERNEL

__global__ void L2_2D_kernel_4(
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

		//L = 0,5f * (Yh - Y)^2
		loss.x = yh.x - y.x; loss.x *= loss.x;
		loss.y = yh.y - y.y; loss.y *= loss.y;
		loss.z = yh.z - y.z; loss.z *= loss.z;
		loss.w = yh.w - y.w; loss.w *= loss.w;

		within_width(loss, index4, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __L2_2D(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { L2_2d_k4_small(stream, Y, Yh, L, lengthv, width, stride); return; }
	L2_2d_k4(stream, 5, 2, Y, Yh, L, lengthv, width, stride);
}

#endif