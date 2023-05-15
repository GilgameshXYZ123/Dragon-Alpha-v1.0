#pragma once

#ifndef SOFTMAX_CROSS_ENTROPY_2D_H
#define SOFTMAX_CROSS_ENTROPY_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SOFTMAX_CROSS_ENTROPY_2D_CALL
#define SOFTMAX_CROSS_ENTROPY_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define softmax_crossEntropy_2d_k4(stream, LB, LT, Y, X, maxX, expXm_max_rowSum, row_lengthv, L, lengthv, width, stride)\
	softmax_crossEntropy_2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, X, maxX, expXm_max_rowSum, row_lengthv, L, lengthv, width, stride)

#define softmax_crossEntropy_2d_k4_small(stream, Y, X, maxX, expXm_max_rowSum, row_lengthv, L, lengthv, width, stride)\
	softmax_crossEntropy_2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, X, maxX, expXm_max_rowSum, row_lengthv, L, lengthv, width, stride)

#endif


#ifndef SOFTMAX_CROSS_ENTROPY_2D_KERNEL
#define SOFTMAX_CROSS_ENTROPY_2D_KERNEL

//Yh = softmax(X) = exp(X - M) / sum(exp(X - M)), let: sum(exp(X - M)) = U
//Yh = exp(X - M) / U
//We have:
//L = crossEntropy(Y, Yh) = -Y*log(Yh) 
//L = -Y * log[exp(X - M) / U]
//L = -Y * {log[exp(X - M)] - logU}
//L = Y * {logU - log(exp(X - M)}
//L = Y * {logU - (X - M)}
//L = Y * {logU - X + M}
//L = Y * {(logU + M) - X}
//As:
//[1] M = maxX
//[2] U = expXm_max_rowSum

__global__ void softmax_crossEntropy_2D_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ X,
	const float* __restrict__ maxX,
	const float* __restrict__ expXm_max_rowSum, int row_lengthv,
	float* __restrict__ L,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 loss;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y = *(float4*)(Y + index4);
		float4 x = *(float4*)(X + index4);

		//[1] M = maxX
		//[2] U = expXm_max_rowSum
		int row_index4 = index4 / row_lengthv;
		float M = maxX[row_index4];
		float U = expXm_max_rowSum[row_index4];
		float K = M + logf(U);

		//L = Y * {(logU + M) - X}
		loss.x = y.x * (K - x.x);
		loss.y = y.y * (K - x.y);
		loss.z = y.z * (K - x.z);
		loss.w = y.w * (K - x.w);

		within_width(loss, index4, stride, width);
		*(float4*)(L + index4) = loss;
	}
}

#endif


void __softmax_crossEntropy_2D(cudaStream_t stream,
	const float* Y,
	const float* X,
	const float* maxX, const float* expXm_max_rowSum, int row_lengthv,
	float* L,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { softmax_crossEntropy_2d_k4_small(stream, Y, X, maxX, expXm_max_rowSum, row_lengthv, L, lengthv, width, stride); return; }
	softmax_crossEntropy_2d_k4(stream, 5, 2, Y, X, maxX, expXm_max_rowSum, row_lengthv, L, lengthv, width, stride);
}

#endif