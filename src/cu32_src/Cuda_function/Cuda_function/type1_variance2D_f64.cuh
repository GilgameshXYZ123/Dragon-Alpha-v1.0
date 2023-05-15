#pragma once

#ifndef VARIANCE_2D_F64_H
#define VARIANCE_2D_F64_H

//(1) lengthv = height * stride
//(2) stride = (width + 3)/4*4
//(3) [lenghtv, stride] % 4 == 0
#ifndef VARIANCE_2D_F64_CALL
#define VARIANCE_2D_F64_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define variance2d_f64_k4_small(stream, X_mean, X_sqmean, X_var, lengthv, width, stride)\
	variance2D_f64_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X_mean, X_sqmean, X_var, lengthv, width, stride)

#define variance2d_f64_k4(stream, LB, LT, X_mean, X_sqmean, X_var, lengthv, width, stride)\
	variance2D_f64_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X_mean, X_sqmean, X_var, lengthv, width, stride)

#endif


#ifndef VARIANCE_2D_F64_KERNEL
#define VARIANCE_2D_F64_KERNEL

//X_var = X_sqmean - X_mean^2
__global__ void variance2D_f64_kernel_4(
	const float* __restrict__ X_mean,
	const float* __restrict__ X_sqmean,
	float* __restrict__ X_var,
	int lengthv, int width, int stride)
{
	const int step = gridDim.x*blockDim.x, step4 = step << 2;
	const int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------
		float4 x_mean = *(float4*)(X_mean + index4);
		float4 x_sqmean = *(float4*)(X_sqmean + index4);

		//compute result-------------------------------------------
		float4 x_var;//X_var = X_sqmean - X_mean^2
		x_var.x = x_sqmean.x - ((double)x_mean.x) * x_mean.x;
		x_var.y = x_sqmean.y - ((double)x_mean.y) * x_mean.y;
		x_var.z = x_sqmean.z - ((double)x_mean.z) * x_mean.z;
		x_var.w = x_sqmean.w - ((double)x_mean.w) * x_mean.w;

		//write data-----------------------------------------------
		within_width(x_var, index4, stride, width);
		*(float4*)(X_var + index4) = x_var;
	}
}

#endif


void __variance2D_f64(cudaStream_t stream,
	const float* X_mean,
	const float* X_sqmean,
	float* X_var,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { variance2d_f64_k4_small(stream, X_mean, X_sqmean, X_var, lengthv, width, stride); return; }
	variance2d_f64_k4(stream, 5, 2, X_mean, X_sqmean, X_var, lengthv, width, stride);
}

#endif