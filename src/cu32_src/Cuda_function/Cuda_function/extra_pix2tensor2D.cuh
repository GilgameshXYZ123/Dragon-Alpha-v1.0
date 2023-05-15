#pragma once

#ifndef PIX_TO_TENSOR_2D_H
#define PIX_TO_TENSOR_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//Time = 0.088 mesc, Speed = 88.7784GB / s
#ifndef PIX_TO_TENSOR_2D_CALL
#define PIX_TO_TENSOR_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define pix2tensor2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	pix2tensor2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define pix2tensor2d_k4_small(stream, X, Y, lengthv, width, stride)\
	pix2tensor2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef PIX_TO_TENSOR_2D_KERNEL
#define PIX_TO_TENSOR_2D_KERNEL

__global__ void pix2tensor2D_kernel_4(
	const char* __restrict__ X, 
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		uchar4 x = *(uchar4*)(X + index4);

		float4 y; 
		y.x = (float)x.x / 255.0f;
		y.y = (float)x.y / 255.0f;
		y.z = (float)x.z / 255.0f;
		y.w = (float)x.w / 255.0f;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __pix2tensor2D(cudaStream_t stream,
	const char* X,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { pix2tensor2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	pix2tensor2d_k4(stream, 5, 2, X, Y, lengthv, width, stride);
}

#endif