#pragma once

#ifndef TANH_2D_H
#define TANH_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef TANH_2D_CALL
#define TANH_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define tanh2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	tanh2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define tanh2d_k4_small(stream, X, Y, lengthv, width, stride)\
	tanh2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef TANH_2D_KERNEL
#define TANH_2D_KERNEL

//[1] Y = tanh(X) = (e^X - e^(-X))/(e^X + e^(-X)) 
//	= {e^(2*X) - 1} / {e^(2*X) + 1}
//	= 1 - 2/(e^(2*X) + 1) 
//[2] Y = tanh(X) =  (e^X - e^(-X))/(e^X + e^(-X))
//	= (1 - e^(-2*X))/(1 + e^(-2*X)) 
//	=  2/(1 + e^(-2*X)) - 1

__global__ void tanh2D_kernel_4(
	const float* __restrict__ X,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		
		float4 y;//VERSION2: Y = 2/(1 + e^(-2*X)) - 1
		y.x = 2.0f / (1.0f + expf(-2.0f * x.x)) - 1.0f;
		y.y = 2.0f / (1.0f + expf(-2.0f * x.y)) - 1.0f;
		y.z = 2.0f / (1.0f + expf(-2.0f * x.z)) - 1.0f;
		y.w = 2.0f / (1.0f + expf(-2.0f * x.w)) - 1.0f;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __tanh2D(cudaStream_t stream,
	const float* X,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { tanh2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	tanh2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif