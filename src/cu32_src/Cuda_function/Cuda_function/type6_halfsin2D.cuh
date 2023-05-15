#pragma once

#ifndef HALF_SIN_2D_H
#define HALF_SIN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef HALF_SIN_2D_CALL
#define HALF_SIN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define halfsin2d_k4(stream, LB, LT, Amp, alpha, X, beta, Y, lengthv, width, stride)\
	halfsin2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Amp, alpha, X, beta, Y, lengthv, width, stride)

#define halfsin2d_k4_small(stream, Amp, alpha, X, beta, Y, lengthv, width, stride)\
	halfsin2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Amp, alpha, X, beta, Y, lengthv, width, stride)

#endif


#ifndef HALF_SIN_2D_KERNEL
#define HALF_SIN_2D_KERNEL

//Y  = Amp * {2*|sin(alpha*X + beta)| - 1}
//Y = 2Amp * {|sin(alpha*X + beta)} - 0.5f}
//Y = 2Amp * |sin(alpha*X + beta)| - Amp
__global__ void halfsin2D_kernel_4(
	float Amp, float alpha, 
	const float* __restrict__ X, float beta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float Amp2 = 2.0f * Amp;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		simdLinear4(x, alpha, x, beta);

		float4 y;//Y = 2Amp * |sin(alpha*X + beta)| - Amp
		y.x = Amp2 * fabsf(sinf(x.x)) - Amp;
		y.y = Amp2 * fabsf(sinf(x.y)) - Amp;
		y.z = Amp2 * fabsf(sinf(x.z)) - Amp;
		y.w = Amp2 * fabsf(sinf(x.w)) - Amp;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __halfsin2D(cudaStream_t stream,
	float Amp, float alpha, const float* X, float beta,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { halfsin2d_k4_small(stream, Amp, alpha, X, beta, Y, lengthv, width, stride); return; }
	halfsin2d_k4(stream, 5, 2, Amp, alpha, X, beta, Y, lengthv, width, stride);
}

#endif