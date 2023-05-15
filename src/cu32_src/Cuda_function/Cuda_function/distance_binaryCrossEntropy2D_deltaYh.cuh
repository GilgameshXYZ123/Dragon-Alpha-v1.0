#pragma once

#ifndef BINARY_CROSS_ENTROPY_2D_DELTAYH_H
#define BINARY_CROSS_ENTROPY_2D_DELTAYH_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef BINARY_CROSS_ENTROPY_2D_DELTAYH_CALL
#define BINARY_CROSS_ENTROPY_2D_DELTAYH_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define binaryCrossEntropy2d_deltaYh_k4_small(stream, Y, Yh, alpha, beta, deltaYh, lengthv, width, stride)\
	binaryCrossEntropy2D_deltaYh_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, Yh, alpha, beta, deltaYh, lengthv, width, stride)

#define binaryCrossEntropy2d_deltaYh_k4(stream, LB, LT, Y, Yh, alpha, beta, deltaYh, lengthv, width, stride)\
	binaryCrossEntropy2D_deltaYh_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, Yh, alpha, beta, deltaYh, lengthv, width, stride)

#endif


#ifndef BINARY_CROSS_ENTROPY_2D_DELTAYH_KERNEL
#define BINARY_CROSS_ENTROPY_2D_DELTAYH_KERNEL

__global__ void binaryCrossEntropy2D_deltaYh_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ Yh,
	float alpha, float beta,
	float* __restrict__ deltaYh,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 dyh, table[2]; table[0] = make_float4(0, 0, 0, 0);
	alpha = -alpha;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 yh = *(float4*)(Yh + index4);
		float4 y  = *(float4*)(Y  + index4);

		//deltaYh = -Y/Yh + (1 - Y)/(1 - Yh)
		dyh.x = alpha * (y.x / yh.x) + beta * (1.0f - y.x) / (1.0f - yh.x);
		dyh.y = alpha * (y.y / yh.y) + beta * (1.0f - y.y) / (1.0f - yh.y);
		dyh.z = alpha * (y.z / yh.z) + beta * (1.0f - y.z) / (1.0f - yh.z);
		dyh.w = alpha * (y.w / yh.w) + beta * (1.0f - y.w) / (1.0f - yh.w);

		within_width_zero_nan(dyh, index4, table, stride, width);
		*(float4*)(deltaYh + index4) = dyh;
	}
}

#endif


void __binaryCrossEntropy_2D_deltaYh(cudaStream_t stream,
	const float* Y, const float* Yh,
	float alpha, float beta,
	float* deltaYh,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { binaryCrossEntropy2d_deltaYh_k4_small(stream, Y, Yh, alpha, beta, deltaYh, lengthv, width, stride); return; }
	binaryCrossEntropy2d_deltaYh_k4(stream, 5, 2, Y, Yh, alpha, beta, deltaYh, lengthv, width, stride);
}

#endif