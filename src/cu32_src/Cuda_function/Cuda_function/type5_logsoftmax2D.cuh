#pragma once

#ifndef LOG_SOFTMAX_2D_H
#define LOG_SOFTMAX_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LOG_SOFTMAX_2D_CALL
#define LOG_SOFTMAX_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define logsoftmax2d_k4(stream, LB, LT, X, maxX, expXm_max_rowSum, row_lengthv, Y, lengthv, width, stride)\
	logsoftmax2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, maxX, expXm_max_rowSum, row_lengthv, Y, lengthv, width, stride)

#define logsoftmax2d_k4_small(stream, X, maxX, expXm_max_rowSum, row_lengthv, Y, lengthv, width, stride)\
	logsoftmax2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, maxX, expXm_max_rowSum, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef LOG_SOFTMAX_2D_KERNEL
#define LOG_SOFTMAX_2D_KERNEL

//Y = logsoftmax(X) = log(softmax(X))
//Y = log[exp(X - M) / U]
//Y = log[exp(X - M)] - log(U)
//Y = (X - M) - log(U)
//Y = X - (M + logU)
//[1] M = maxX
//[2] U = expXm_max_rowSum

__global__ void logsoftmax2D_kernel_4(
	const float* __restrict__ X,
	const float* maxX,
	const float* __restrict__ expXm_max_rowSum, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		
		//[1] M = maxX
		//[2] U = expXm_max_rowSum
		int row_index4 = index4 / row_lengthv;
		float M = maxX            [row_index4];
		float U = expXm_max_rowSum[row_index4];
		float K = M + logf(U);

		float4 y;//Y = (X - M) - log(U)
		y.x = x.x - K;
		y.y = x.y - K;
		y.z = x.z - K;
		y.w = x.w - K;

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __logsoftmax2D(cudaStream_t stream,
	const float* X,
	const float* maxX,
	const float* expXm_max_rowSum, int row_lengthv,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { logsoftmax2d_k4_small(stream, X, maxX, expXm_max_rowSum, row_lengthv, Y, lengthv, width, stride); return; }
	logsoftmax2d_k4(stream, 5, 2, X, maxX, expXm_max_rowSum, row_lengthv, Y, lengthv, width, stride);
}

#endif