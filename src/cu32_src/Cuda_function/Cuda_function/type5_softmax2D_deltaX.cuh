#pragma once

#ifndef SOFTMAX_2D_DELTAX_H
#define SOFTMAX_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef SOFTMAX_2D_DELTAX_CALL
#define SOFTMAX_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define softmax2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, deltaY_Y_rowSum, row_lengthv,lengthv, width, stride)\
	softmax2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, deltaY_Y_rowSum, row_lengthv, lengthv, width, stride)

#define softmax2d_deltaX_k4_small(stream,  deltaX, deltaY, Y, deltaY_Y_rowSum, row_lengthv, lengthv, width, stride)\
	softmax2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, deltaY_Y_rowSum, row_lengthv, lengthv, width, stride)

#endif


#ifndef SOFTMAX_2D_DELTAX_KERNEL
#define SOFTMAX_2D_DELTAX_KERNEL

//Y = softmax(X)
//deltaX[k] = deltaY[k] * d"Y[k]"/d"X[k]" + 
//		sum_row(i!=k, deltaY[i] * d"Y[i]"/d"X[k]")
//deltaX[k] = deltaY[k] * Y[k] * (1 - Y[k]) + 
//		sum_row(i!=k, deltaY[i] * -Y[i] * Y[k]) 
//		+ (deltaY[k] * -Y[k] * Y[k])
//		- (deltaY[k] * -Y[k] * Y[k])
//deltaX[k] = deltaY[k] * Y[k] * (1 - Y[k]) + 
//		sum_row(deltaY[i] * -Y[i] * Y[k]) 
//		+ (deltaY[k] * Y[k] * Y[k])
//deltaX[k] = deltaY[k] * Y[k] - Y[k] * sum_row(deltaY[i]*Y[i]) 
//deltaX[k] = Y[k] * {deltaY[k] - sum_row(deltaY[i]*Y[i])}
//deltaX = Y * {deltaY - S}
//[1] S = sum_row(deltaY[i]*Y[i])
//
//Expecially:
//deltaX[k] = Y[k] * {-P[k]/Y[k] + sum_row(P[i]/Y[i] * Y[i])}
//deltaX[k] = Y[k] * {-P[k]/Y[k] + sum_row(P[i])}
//As: sum_row(P[i]) = 1
//deltaX[k] = Y[k] * {-P[k]/Y[k] + 1}
//deltaX[k] = Y[k] - P[k], P is the label

__global__ void softmax2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	const float* __restrict__ deltaY_Y_rowSum, int row_lengthv,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y  = *(float4*)(Y      + index4);
		float4 dy = *(float4*)(deltaY + index4);
		
		//[1] S = sum_row(deltaY[i]*Y[i])
		float S = deltaY_Y_rowSum[index4 / row_lengthv];

		float4 dx;//deltaX = Y * {deltaY - S}
		dx.x = y.x * (dy.x - S);
		dx.y = y.y * (dy.y - S);
		dx.z = y.z * (dy.z - S);
		dx.w = y.w * (dy.w - S);
	
		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __softmax2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *Y,
	const float* deltaY_Y_rowSum, int row_lengthv,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { softmax2d_deltaX_k4_small(stream, deltaX, deltaY, Y, deltaY_Y_rowSum, row_lengthv, lengthv, width, stride); return; }
	softmax2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, deltaY_Y_rowSum, row_lengthv, lengthv, width, stride);
}

#endif