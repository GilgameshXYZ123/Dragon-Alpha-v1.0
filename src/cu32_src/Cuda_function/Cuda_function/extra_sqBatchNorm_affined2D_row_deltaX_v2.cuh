#pragma once

#ifndef SQUARE_BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_H
#define SQUARE_BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
//(4) V2: holdX(), X is not changed
//(5) affined = true
#ifndef SQUARE_BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_CALL
#define SQUARE_BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define sqBatchNorm_affined2d_row_deltaX_v2_k4(stream, LB, LT, deltaY, X, X_mean, X_sqmean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	sqBatchNorm_affined2D_row_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaY, X, X_mean, X_sqmean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#define sqBatchnorm_affined2d_row_deltaX_v2_k4_small(stream, deltaY, X, X_mean, X_sqmean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)\
	sqBatchNorm_affined2D_row_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaY, X, X_mean, X_sqmean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride)

#endif


#ifndef SQUARE_BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_KERNEL
#define SQUARE_BATCH_NORM_AFFINED_2D_ROW_DELTAX_V2_KERNEL

//=======[Document]==================================================
//<1> N = batch_size =  field_length = lengthv / row_lengthv
//<2> M = row_lengthv
//<3> dims: Y[N, M], X[N, M], X_mean[M], X_sqmean[M], A[M], B[M]
//<4> dims: deltaY[N, M], deltaX[N, M], deltaXp1[M], deltaXp2[M]
//
//[Forward Propagation]:
//(1) X_mean = mean_each_field(X)
//(2) X_sqmean = mean_each_field(X^2)
//(3) X_std = sqrt(X_sqmean - X_mean^2 + eps)
//(4) X_norm = (X - X_mean) / X_std
//(5) Y = A * X_norm + B
//
//[Backward Propagation]
//(1) (deltaXp1 = deltaB) = sum_each_field: deltaY
//(2) (deltaXp2 = deltaA) = sum_each_field: deltaY * X_norm
//(3) X_rstd = 1 / X_std
//(4) deltaX = (A * X_rstd) * (deltaY - deltaXp1 / N - deltaXp2 * X_norm / N)
//    deltaX = (A * X_rstd) * (deltaY - (deltaXp1 + deltaXp2 * X_norm) / N)
//STEP:
//(1) rN = (1.0f / N)
//(2) X_rstd = rsqrtf(X_sqmean - X_mean^2 + eps)
//(3) X_norm = (X - X_mean) * X_rstd
//(4) deltaX = (A*X_rstd) * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
//=======[Document]==================================================

__global__ void sqBatchNorm_affined2D_row_deltaX_v2_kernel_4(
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	const float* __restrict__ X_mean,
	const float* __restrict__ X_sqmean, float eps,
	const float* __restrict__ A,
	const float* __restrict__ deltaXp1,
	const float* __restrict__ deltaXp2, int row_lengthv,
	float* __restrict__ deltaX,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;
	
	const float rN = (1.0f * row_lengthv) / lengthv;//(1) rN = (1.0f / N)
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------------------
		const int field_index4 = index4 % row_lengthv;
		const float4 dy = *(float4*)(deltaY + index4);
		const float4 x = *(float4*)(X + index4);
		const float4 x_mean = *(float4*)(X_mean + field_index4);
		const float4 x_sqmean = *(float4*)(X_sqmean + field_index4);
		const float4 a = *(float4*)(A + field_index4);  
		const float4 dxp1 = *(float4*)(deltaXp1 + field_index4);
		const float4 dxp2 = *(float4*)(deltaXp2 + field_index4);

		//compute result-------------------------------------------------------
		float4 x_rstd;//(2) X_rstd = rsqrtf(X_sqmean - X_mean^2 + eps)
		x_rstd.x = rsqrtf(x_sqmean.x - x_mean.x*x_mean.x + eps);
		x_rstd.y = rsqrtf(x_sqmean.y - x_mean.y*x_mean.y + eps);
		x_rstd.z = rsqrtf(x_sqmean.z - x_mean.z*x_mean.z + eps);
		x_rstd.w = rsqrtf(x_sqmean.w - x_mean.w*x_mean.w + eps);

		float4 x_norm;//(3) X_norm = (X - X_mean) * X_rstd
		x_norm.x = (x.x - x_mean.x) * x_rstd.x;
		x_norm.y = (x.y - x_mean.y) * x_rstd.y;
		x_norm.z = (x.z - x_mean.z) * x_rstd.z;
		x_norm.w = (x.w - x_mean.w) * x_rstd.w;

		float4 dx;//<4> deltaX = (A*X_rstd) * (deltaY - rN*(deltaXp1 + deltaXp2*X_norm))
		dx.x = (a.x * x_rstd.x) * (dy.x - rN * (dxp1.x + dxp2.x * x_norm.x));
		dx.y = (a.y * x_rstd.y) * (dy.y - rN * (dxp1.y + dxp2.y * x_norm.y));
		dx.z = (a.z * x_rstd.z) * (dy.z - rN * (dxp1.z + dxp2.z * x_norm.z));
		dx.w = (a.w * x_rstd.w) * (dy.w - rN * (dxp1.w + dxp2.w * x_norm.w));

		//write data-----------------------------------------------------------
		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __sqBatchNorm_affined2D_row_deltaX_v2(cudaStream_t stream,
	const float* deltaY, const float* X,
	const float* X_mean,
	const float* X_sqmean, float eps,
	const float* A,
	const float* deltaXp1,
	const float* deltaXp2, int row_lengthv,
	float* deltaX,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { sqBatchnorm_affined2d_row_deltaX_v2_k4_small(stream, deltaY, X, X_mean, X_sqmean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride); return; }
	sqBatchNorm_affined2d_row_deltaX_v2_k4(stream, 5, 2, deltaY, X, X_mean, X_sqmean, eps, A, deltaXp1, deltaXp2, row_lengthv, deltaX, lengthv, width, stride);
}

#endif