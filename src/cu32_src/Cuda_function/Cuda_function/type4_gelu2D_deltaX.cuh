#pragma once


#ifndef GELU_2D_DELTAX_H
#define FELU_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef GELU_2D_DELTAX_CALL
#define FELU_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define gelu2d_deltaX_k4(stream, LB, LT, deltaX, deltaY, Y, lengthv, width, stride)\
	gelu2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

#define gelu2d_deltaX_k4_small(stream,  deltaX, deltaY, Y, lengthv, width, stride)\
	gelu2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, lengthv, width, stride)

#endif

//y = x * sigmoid(1.702x)
//let: a = 1.702
//y = x * sigmoid(a*x)
//y' = sigmoid(a*x) + x*sigmoid'(a*x)
//y' = sigmoid(ax) + a*[x * sigmoid(ax)] * (1 - sigmoid(ax))
//As: y = x * sigmoid(ax), 
//y' = sigmoid(ax) + a*(1 - sigmoid(ax)) * y
//As: sigmoid(ax) = y/x
//y' = y/x + a*(1 - y/x)*y

__global__ void gelu2D_deltaX_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 y = *(float4*)(&Y[index4]);

		float4 dx;//find derivative
		
		float4 dy = *(float4*)(&deltaY[index4]);
		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(&deltaX[index4]) = dx;
	}
}

void __gelu2D_deltaX(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float *Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { gelu2d_deltaX_k4_small(stream, deltaX, deltaY, Y, lengthv, width, stride); return; }
	gelu2d_deltaX_k4(stream, 5, 2, deltaX, deltaY, Y, lengthv, width, stride);
}


#endif