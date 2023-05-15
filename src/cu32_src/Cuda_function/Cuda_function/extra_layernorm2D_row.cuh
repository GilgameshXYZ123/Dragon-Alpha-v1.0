#pragma once

#ifndef LAYER_NORM_2D_ROW_H
#define LAYER_NORM_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef LAYER_NORM_2D_ROW_CALL
#define LAYER_NORM_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define layernorm2d_row_k4(stream, LB, LT, X, X_row_mean, X_row_square_mean, eps, row_lengthv, Y, lengthv, width, stride)\
	layernorm2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X_row_mean, X_row_square_mean, eps, row_lengthv, Y,lengthv, width, stride)

#define layernorm2d_row_k4_small(stream, X, X_row_mean, X_row_square_mean, eps, row_lengthv, Y, lengthv, width, stride)\
	layernorm2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X_row_mean, X_row_square_mean, eps, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef LAYER_NORM_2D_ROW_KERNEL
#define LAYER_NORM_2D_ROW_KERNEL

//X2_lengthv = X_mean.lengthv = X_square_mean.lengthv
__global__ void layernorm2D_row_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ X_row_mean,//[field_length/column_length]
	const float* __restrict__ X_square_mean, float eps,//[field_length/colum_length]
	int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x = *(float4*)(X + index4);

		int row_index4   = index4 / row_lengthv;
		//int field_index4 = index4 - row_index4 * row_lengthv;//index4 % row_lengthv;

		float x_mean        = X_row_mean   [row_index4];
		float x_square_mean = X_square_mean[row_index4];

		//X_stddev = sqrt(X_square_mean - X_mean*X_mean + eps)
		float x_stddev = rsqrtf(x_square_mean - x_mean * x_mean + eps);

		float4 y;//Y[i] = (X[i] - X_mean) / X_stddev
		y.x = (x.x - x_mean) * x_stddev; 
		y.y = (x.y - x_mean) * x_stddev; 
		y.z = (x.z - x_mean) * x_stddev; 
		y.w = (x.w - x_mean) * x_stddev; 

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __layernorm2D_row(cudaStream_t stream,
	const float* X,
	const float* X_row_mean,
	const float* X_row_square_mean, float eps, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { layernorm2d_row_k4_small(stream, X, X_row_mean, X_row_square_mean, eps, row_lengthv, Y, lengthv, width, stride); return; }
	layernorm2d_row_k4(stream, 5, 2, X, X_row_mean, X_row_square_mean, eps, row_lengthv, Y, lengthv, width, stride);
}

#endif