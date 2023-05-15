#pragma once

#ifndef SOFTMAX_CROSS_ENTROPY_2D_DELTAX_H
#define SOFTMAX_CROSS_ENTROPY_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SOFTMAX_CROSS_ENTROPY_2D_DELTAX_CALL
#define SOFTMAX_CROSS_ENTROPY_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define softmax_crossEntropy_2d_deltaX_k4(stream, LB, LT, Y, X, maxX, expXm_max_rowSum, Y_rowSum, row_lengthv, deltaX, lengthv, width, stride)\
	softmax_crossEntropy_2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(Y, X, maxX, expXm_max_rowSum, Y_rowSum, row_lengthv, deltaX, lengthv, width, stride)

#define softmax_crossEntropy_2d_deltaX_k4_small(stream, Y, X, maxX, expXm_max_rowSum, Y_rowSum, row_lengthv, deltaX, lengthv, width, stride)\
	softmax_crossEntropy_2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(Y, X, maxX, expXm_max_rowSum, Y_rowSum, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef SOFTMAX_CROSS_ENTROPY_2D_DELTAX_KERNEL
#define SOFTMAX_CROSS_ENTROPY_2D_DELTAX_KENREL

//(1) Yh = softmax(X), L = crossEntropy(Y, Yh) = sum: -Y*log(Yh)
//(1) deltaYh = -Y / Yh
//we have:
//deltaX = deltaYh * Yh'
//deltaX = Yh * {deltaYh - sum_row(deltaYh[i]*Yh[i])}
//deltaX = Yh * { -Y / Yh - sum_row(-Y[i] / Yh[i] * Yh[i])}
//deltaX = Yh * { -Y / Yh + sum_row(Y[i]) }
//deltaX = -Y + Yh * sum_row(Y[i])
//deltaX = -Y + softmax(X) * sum_row(Y[i])
//deltaX = -Y + exp(X - M) / U * S
//deltaX = -Y + exp(X - M) * (S / U)
//[1] S = sum_row(Y[i])
//[2] M = maxX, for each row
//[3] U = expXm_max_rowSum, for each_row
//especially: when sum_row(Y[i]) == 1
//we have: deltaX = Yh - Y = softmax(X) - Y

__global__ void softmax_crossEntropy_2D_deltaX_kernel_4(
	const float* __restrict__ Y,
	const float* __restrict__ X,
	const float* __restrict__ maxX,
	const float* __restrict__ expXm_max_rowSum, 
	const float* __restrict__ Y_rowSum, int row_lengthv,
	float* __restrict__ deltaX,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y = *(float4*)(Y + index4);
		float4 x = *(float4*)(X + index4);

		int row_index4 = index4 / row_lengthv;
		float S = Y_rowSum        [row_index4];//[1] S = sum_row(Y[i])
		float M = maxX            [row_index4];//[2] M = maxX, for each row
		float U = expXm_max_rowSum[row_index4];//[3] U = expXm_max_rowSum, for each_row
		float G = S / U;

		float4 dx;//deltaX = -Y + exp(X - M) * (S / U)
		dx.x = -y.x + expf(x.x - M) * G;
		dx.y = -y.y + expf(x.y - M) * G;
		dx.z = -y.z + expf(x.z - M) * G;
		dx.w = -y.w + expf(x.w - M) * G;

		within_width_zero_nan(dx, index4, table, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __softmax_crossEntropy_2D_deltaX(cudaStream_t stream,
	const float* Y,
	const float* X,
	const float* maxX, 
	const float* expXm_max_rowSum,
	const float* Y_rowSum, int row_lengthv,
	float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { softmax_crossEntropy_2d_deltaX_k4_small(stream, Y, X, maxX, expXm_max_rowSum, Y_rowSum, row_lengthv, deltaX, lengthv, width, stride); return; }
	softmax_crossEntropy_2d_deltaX_k4(stream, 5, 2, Y, X, maxX, expXm_max_rowSum, Y_rowSum, row_lengthv, deltaX, lengthv, width, stride);
}

#endif