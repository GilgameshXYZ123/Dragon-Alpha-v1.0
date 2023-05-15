#pragma once

#ifndef ONE_HOT_2D_ROW_INT_H
#define ONE_HOT_2D_ROW_INT_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lengthv, row_lengthv] % stride == 0
//field_length * row_lengthv = X.lengthv
#ifndef ONE_HOT_2D_ROW_INT_CALL
#define ONE_HOT_2D_ROW_INT_CALL

#define onehot2d_row_int_k4_small(stream, X, alpha, beta, row_lengthv, Y, lengthv, width, stride)\
	onehot2D_row_int_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, alpha, beta, row_lengthv, Y, lengthv, width, stride)

#define onehot2d_row_int_k4(stream, LB, LT, X, alpha, beta, row_lengthv, Y, lengthv, width, stride)\
	onehot2D_row_int_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, alpha, beta, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef ONE_HOT_2D_ROW_INT_KERNEL
#define ONE_HOT_2D_ROW_INT_KERNEL

//[N, IW] = [field_length, row_lengthv]
//Y = tensor2D[N, IW]
//X = tensor1D[N]
//for each row[i](1:N): 
//	Y[i] = (field_index == X[i]? alpha, beta)

__global__ void onehot2D_row_int_kernel_4(
	const int* __restrict__ X,
	float alpha, float beta, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		int field_index4 = index4 / row_lengthv;
		int x = X[field_index4];

		float4 row_index4;//index4 % row_lengthv;
		row_index4.x = index4 - field_index4 * row_lengthv;
		row_index4.y = row_index4.x + 1;
		row_index4.z = row_index4.x + 2;
		row_index4.w = row_index4.x + 3;

		float4 y;
		y.x = (row_index4.x == x)*alpha + (row_index4.x != x) * beta;
		y.y = (row_index4.y == x)*alpha + (row_index4.y != x) * beta;
		y.z = (row_index4.z == x)*alpha + (row_index4.z != x) * beta;
		y.w = (row_index4.w == x)*alpha + (row_index4.w != x) * beta;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __onehot2D_row_int(cudaStream_t stream,
	const int* X,
	float alpha, float beta, int row_lengthv,
	float *Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { onehot2d_row_int_k4_small(stream, X, alpha, beta, row_lengthv, Y, lengthv, width, stride); return; }
	onehot2d_row_int_k4(stream, 5, 2, X, alpha, beta, row_lengthv, Y, lengthv, width, stride);
}

#endif
