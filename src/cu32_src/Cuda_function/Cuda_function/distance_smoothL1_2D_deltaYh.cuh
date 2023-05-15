#pragma once

#ifndef SMOOTH_L1_2D_DELTAYH_H
#define SMOOTH_L1_2D_DELTAYH_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SMOOTH_L1_2D_DELTAYH_CALL
#define SMOOTH_L1_2D_DELTAYH_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define smoothL1_2d_deltaYh_k4(stream, LB, LT, Y, Yh, deltaYh, lengthv, width, stride)\
	smoothL1_2D_deltaYh_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, deltaYh, lengthv, width, stride)

#define smoothL1_2d_deltaYh_k4_small(stream, Y, Yh, deltaYh, lengthv, width, stride)\
	smoothL1_2D_deltaYh_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, deltaYh, lengthv, width, stride)

#endif


#ifndef SMOOTH_L1_2D_DELTAYH_KERNEL
#define SMOOTH_L1_2D_DELTAYH_KERNEL

__global__ void smoothL1_2D_deltaYh_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float* __restrict__ deltaYh,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 dyh, div, sign;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y = *(float4*)(Y + index4);

		//deltaYh = sign(Yh - Y)*{(div<=1)*(div - 1) + 1}
		div.x = yh.x - y.x; sign.x = SIGN(div.x); div.x = fabsf(div.x);
		div.y = yh.y - y.y; sign.y = SIGN(div.y); div.y = fabsf(div.y);
		div.z = yh.z - y.z; sign.z = SIGN(div.z); div.z = fabsf(div.z);
		div.w = yh.w - y.w; sign.w = SIGN(div.w); div.w = fabsf(div.w);

		dyh.x = sign.x * ((div.x <= 1) * (div.x - 1.0f) + 1.0f);
		dyh.y = sign.y * ((div.y <= 1) * (div.y - 1.0f) + 1.0f);
		dyh.z = sign.z * ((div.z <= 1) * (div.z - 1.0f) + 1.0f);
		dyh.w = sign.w * ((div.w <= 1) * (div.w - 1.0f) + 1.0f);

		within_width(dyh, index4, stride, width);
		*(float4*)(deltaYh + index4) = dyh;
	}
}

#endif


void __smoothL1_2D_deltaYh(cudaStream_t stream,
	const float* Y, const float* Yh,
	float* deltaYh,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { smoothL1_2d_deltaYh_k4_small(stream, Y, Yh, deltaYh, lengthv, width, stride); return; }
	smoothL1_2d_deltaYh_k4(stream, 5, 2, Y, Yh, deltaYh, lengthv, width, stride);
}

#endif