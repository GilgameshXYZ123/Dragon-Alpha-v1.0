#pragma once

#ifndef AFFINE_2D_ROW_H
#define AFFINE_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef AFFINE_2D_ROW_CALL
#define AFFINE_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define affine2d_row_k4(stream, LB, LT, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride)

#define affine2d_row_k4_small(stream, X, A, B, row_lengthv, Y, lengthv, width, stride)\
	affine2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, A, B, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef AFFINE_2D_ROW_KERNEL
#define AFFINE_2D_ROW_KERNEL

//X2_lengthv = X_mean.lengthv = X_square_mean.lengthv
__global__ void affine2D_row_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		int field_index4 = index4 % row_lengthv;
		float4 a = *(float4*)(A + field_index4);
		float4 b = *(float4*)(B + field_index4);

		float4 y;//Y[i] = A * X[i] + B
		y.x = (a.x * x.x) + b.x;
		y.y = (a.y * x.y) + b.y;
		y.z = (a.z * x.z) + b.z;
		y.w = (a.w * x.w) + b.w;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __affine2D_row(cudaStream_t stream,
	const float* X,
	const float* A,
	const float* B, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { affine2d_row_k4_small(stream, X, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	affine2d_row_k4(stream, 5, 2, X, A, B, row_lengthv, Y, lengthv, width, stride);
}

#endif