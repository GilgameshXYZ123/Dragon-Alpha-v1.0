#pragma once

#ifndef SIGMOID_BINARY_CROSS_ENTROPY_2D_DELTAX_H
#define SIGMOID_BINARY_CROSS_ENTROPY_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SIGMOID_BINARY_CROSS_ENTROPY_2D_DELTAX_CALL
#define SIGMOID_BINARY_CROSS_ENTROPY_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sigmoid_binaryCrossEntropy2d_deltaX_k4(stream, LB, LT, Y, X, alpha, beta, deltaX, lengthv, width, stride)\
	sigmoid_binaryCrossEntropy2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, X, alpha, beta, deltaX, lengthv, width, stride)

#define sigmoid_binaryCrossEntropy2d_deltaX_k4_small(stream, Y, X, alpha, beta, deltaX, lengthv, width, stride)\
	sigmoid_binaryCrossEntropy2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, X, alpha, beta, deltaX, lengthv, width, stride)

#endif


#ifndef SIGMOID_BINARY_CROSS_ENTROPY_2D_DELTAX_KERNEL
#define SIGMOID_BINARY_CROSS_ENTROPY_2D_DELTAX_KERNEL

//[1] deltaYh = -alpha * Y / Yh + beta * (1 - Y) / (1 - Yh)
//[2] deriYh = Yh * (1 - Yh)
//We have: 
//deltaX = deriYh * deltaYh
//deltaX = Yh * (1 - Yh) * {-alpha * Y / Yh + beta * (1 - Y) / (1 - Yh)}
//deltaX = alpha * Y * (Yh - 1) + beta * (1 - Y) * Yh
//deltaX = alpha * {Y*Yh - Y} + beta * (Yh - Y*Yh)
//deltaX = (alpha - beta)*Y*Yh + (beta*Yh - alpha*Y)
//deltaX = Yh * {(alpha - beta)*Y + beta} - alpha*Y
//As: Yh = 1/(1 + exp(-X))
//deltaX = {(alpha - beta)*Y + beta} /(1 + exp(-X)) - alpha*Y
//when: alpha = beta = 1
//deltaX = Yh - Y

__global__ void sigmoid_binaryCrossEntropy2D_deltaX_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ X,
	float alpha, float beta,
	float* __restrict__ deltaX,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	float gamma = alpha - beta;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y = *(float4*)(Y + index4);
		float4 x = *(float4*)(X + index4);

		float4 dx;//deltaX = {(alpha - beta)*Y + beta} / (1 + exp(-X)) - alpha*Y
		dx.x = (gamma*y.x + beta) / (1.0f + expf(-x.x)) - alpha * y.x;
		dx.y = (gamma*y.y + beta) / (1.0f + expf(-x.y)) - alpha * y.y;
		dx.z = (gamma*y.z + beta) / (1.0f + expf(-x.z)) - alpha * y.z;
		dx.w = (gamma*y.w + beta) / (1.0f + expf(-x.w)) - alpha * y.w;

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __sigmoid_crossEntropy_2D_deltaX(cudaStream_t stream,
	const float* Y, const float* X,
	float alpha, float beta,
	float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sigmoid_binaryCrossEntropy2d_deltaX_k4_small(stream, Y, X, alpha, beta, deltaX, lengthv, width, stride); return; }
	sigmoid_binaryCrossEntropy2d_deltaX_k4(stream, 5, 2, Y, X, alpha, beta, deltaX, lengthv, width, stride);
}

#endif