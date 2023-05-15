#pragma once

#ifndef ZERO_NAN_2D_H
#define ZERO_NAN_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ZERO_NAN_2D_CALL
#define ZERO_NAN_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define zero_nan2d_k4(stream, LB, LT, X, Y, lengthv, width, stride)\
	zero_nan2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#define zero_nan2d_k4_small(stream, X, Y, lengthv, width, stride)\
	zero_nan2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, Y, lengthv, width, stride)

#endif


#ifndef ZERO_NAN_2D_KERNEL
#define ZERO_NAN_2D_KERNEL

__global__ void zero_nan2D_kernel_4(
	const float* __restrict__ X, 
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//if nan: (x == x) == 0, table[0] = make_float(0, 0, 0, 0)
	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = table[1] = *(float4*)(X + index4);
	
		float4 y; 
		y.x = table[x.x == x.x].x;
		y.y = table[x.y == x.y].y;
		y.z = table[x.z == x.z].z;
		y.w = table[x.w == x.w].w;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __zero_nan2D(cudaStream_t stream,
	const float* X, float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { zero_nan2d_k4_small(stream, X, Y, lengthv, width, stride); return; }
	zero_nan2d_k4(stream, 5, 2,  X, Y, lengthv, width, stride);
}

#endif