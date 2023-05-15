#pragma once

#ifndef SQUARE_BATCH_NORM_AFFINED_2D_ROW_H
#define SQUARE_BATCH_NORM_AFFINED_2D_ROW_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) affined = true
#ifndef SQUARE_BATCH_NORM_AFFINED_2D_ROW_CALL
#define SQUARE_BATCH_NORM_AFFINED_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sqBatchNorm_affined2d_row_k4(stream, LB, LT, X, X_mean, X_sqmean, eps, A, B, row_lengthv, Y, lengthv, width, stride)\
	sqBatchNorm_affined2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X, X_mean, X_sqmean, eps, A, B, row_lengthv, Y, lengthv, width, stride)

#define sqBatchNorm_affined2d_row_k4_small(stream, X, X_mean, X_sqmean, eps, A, B, row_lengthv, Y, lengthv, width, stride)\
	sqBatchNorm_affined2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X, X_mean, X_sqmean, eps, A, B, row_lengthv, Y, lengthv, width, stride)

#endif


#ifndef SQUARE_BATCH_NORM_AFFINED_2D_ROW_KERNEL
#define SQUARE_BATCH_NORM_AFFINED_2D_ROW_KERNEL

//=======[Document]==================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_sqmean[M], A[M], B[M]
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_sqmean = mean_each_field(X^2)
//(3) X_std = sqrt(X_sqmean - X_mean^2 + eps)
//(4) X_norm = (X - X_mean) / X_std
//(5) Y = A * X_norm + B
//=======[Document]==================================================

__global__ void sqBatchNorm_affined2D_row_kernel_4(
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_sqmean, float eps,
	const float* __restrict__ A,
	const float* __restrict__ B, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------
		int field_index4 = index4 % row_lengthv;
		float4 x = *(float4*)(X + index4);
		float4 x_mean = *(float4*)(X_mean + field_index4);
		float4 x_sqmean = *(float4*)(X_sqmean + field_index4);
		float4 a = *(float4*)(A + field_index4);
		float4 b = *(float4*)(B + field_index4);

		//compute result-------------------------------------------------
		float4 x_norm;//<1> X_norm = (X - X_mean) / sqrt(X_sqmean - X_mean^2 + eps)
		x_norm.x = (x.x - x_mean.x) * rsqrtf(x_sqmean.x - x_mean.x * x_mean.x + eps);
		x_norm.y = (x.y - x_mean.y) * rsqrtf(x_sqmean.y - x_mean.y * x_mean.y + eps);
		x_norm.z = (x.z - x_mean.z) * rsqrtf(x_sqmean.z - x_mean.z * x_mean.z + eps);
		x_norm.w = (x.w - x_mean.w) * rsqrtf(x_sqmean.w - x_mean.w * x_mean.w + eps);

		float4 y;//A * X_norm + B
		y.x = a.x * x_norm.x + b.x;
		y.y = a.y * x_norm.y + b.y;
		y.z = a.z * x_norm.z + b.z;
		y.w = a.w * x_norm.w + b.w;
		
		//write data-----------------------------------------------------
		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __sqBatchNorm_affined2D_row(cudaStream_t stream,
	const float* X,
	const float* X_mean,
	const float* X_sqmean, float eps,
	const float* A,
	const float* B, int row_lengthv,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sqBatchNorm_affined2d_row_k4_small(stream, X, X_mean, X_sqmean, eps, A, B, row_lengthv, Y, lengthv, width, stride); return; }
	sqBatchNorm_affined2d_row_k4(stream, 5, 2, X, X_mean, X_sqmean, eps, A, B, row_lengthv, Y, lengthv, width, stride);
}

#endif