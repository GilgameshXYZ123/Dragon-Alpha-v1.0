#pragma once

#ifndef ELU_2D_H
#define ELU_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ELU_2D_CALL
#define ELU_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define elu2d_k4(stream, LB, LT,  X, alpha, k, Y, lengthv, width, stride)\
	elu2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, k, Y, lengthv, width, stride)

#define elu2d_k4_small(stream, X, alpha, k, Y, lengthv, width, stride)\
	elu2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, alpha, k, Y, lengthv, width, stride)

#endif


#ifndef ELU_2D_KERNEL
#define ELU_2D_KERNEL

#define ELU(x, k) ((x>0)*x + (x<0)*k*(expf(x)-1.0f))

__global__ void elu2D_kernel_4(
	const float* __restrict__ X, float alpha, float k,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);
		
		float4 y; 
		y.x = alpha * ELU(x.x, k);
		y.y = alpha * ELU(x.y, k);
		y.z = alpha * ELU(x.z, k);
		y.w = alpha * ELU(x.w, k);

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __elu2D(cudaStream_t stream,
	const float* X, float alpha, float k,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { elu2d_k4_small(stream, X, alpha, k, Y, lengthv, width, stride); return; }
	elu2d_k4(stream, 5, 2, X, alpha, k, Y, lengthv, width, stride);
}

#endif