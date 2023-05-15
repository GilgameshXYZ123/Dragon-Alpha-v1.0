#pragma once

#ifndef LOG_SOFTMAX_2D_DELTAX_H
#define LOG_SOFTMAX_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LOG_SOFTMAX_2D_DELTAX_CALL
#define LOG_SOFTMAX_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define logsoftmax2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, deltaY_rowSum, row_lengthv,lengthv, width, stride)\
	logsoftmax2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, deltaY_rowSum, row_lengthv, lengthv, width, stride)

#define logsoftmax2d_deltaX_k4_small(stream,  deltaX, deltaY, Y, deltaY_rowSum, row_lengthv, lengthv, width, stride)\
	logsoftmax2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, deltaY_rowSum, row_lengthv, lengthv, width, stride)

#endif


#ifndef LOG_SOFTMAX_2D_DELTAX_KERNEL
#define LOG_SOFTMAX_2D_DELTAX_KERNEL

//(1) Y = log(Z), Z = softmax(X)
//(2) deltaZ = deltaY * Y' = deltaY / Z
//(3) Z = expf(Y)
//we have:
//deltaX = deltaZ * Z' 
//deltaX = Z * { deltaZ - sum_row(deltaZ[i]*Z[i]) }
//deltaX = Z * { deltaY / Z - sum_row(deltaY[i] / Z[i] * Z[i]) }
//deltaX = Z * { deltaY / Z - sum_row(deltaY[i]) }
//deltaX = deltaY - Z * sum_row(deltaY[i])
//deltaX = deltaY - expf(Y) * sum_row(deltaY[i])
//deltaX = {deltaY - exp(Y) * S}
//[1] S = sum_row(deltaY[i]) = deltaY_row_sum

__global__ void logsoftmax2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ deltaY_rowSum, int row_lengthv,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);

		//[1] S = sum_row(deltaY[i]) = deltaY_row_sum
		float S = deltaY_rowSum[index4 / row_lengthv];

		float4 dx;//deltaX = {deltaY - exp(Y) * S}
		dx.x = dy.x - expf(y.x)*S;
		dx.y = dy.y - expf(y.y)*S;
		dx.z = dy.z - expf(y.z)*S;
		dx.w = dy.w - expf(y.w)*S;
		
		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __logsoftmax2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* Y,
	const float* deltaY_rowSum, int row_lengthv,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { logsoftmax2d_deltaX_k4_small(stream, deltaX, deltaY, Y, deltaY_rowSum, row_lengthv, lengthv, width, stride); return; }
	logsoftmax2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, deltaY_rowSum, row_lengthv, lengthv, width, stride);
}

#endif