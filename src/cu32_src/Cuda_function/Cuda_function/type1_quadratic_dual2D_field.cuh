#pragma once

#ifndef QUADRATIC_DUAL_2D_FIELD_H
#define QUADRATIC_DUAL_2D_FIELD_H

//lengthv = height * stride
//stride = (width + 3)/4 * 4, stride >= 4, stride % 4 ==0
//[lengthv, row_lengthv] % stride == 0
//X2 must a 1D Tensor[field_length]
//field_length * row_lengthv = X1.lengthv
#ifndef QUADRATIC_DUAL_2D_FIELD_CALL
#define QUADRATIC_DUAL_2D_FIELD_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define quadratic_dual2d_field_k4_small(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_field_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

#define quadratic_dual2d_field_k4(stream, LB, LT, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)\
	quadratic_dual2D_field_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride)

#endif


#ifndef QUADRATIC_DUAL_2D_FIELD_KERNEL
#define QUADRATIC_DUAL_2D_FIELD_KERNEL

//row_index * row_lengthv + field_index = index4
//row_lengthv = 4 * k, index4 = 4 * index, u belongs to [0, 1, 2, 3]
//(index4 + u) / row_lengthv = (4 * index + u) / (4 * k), 
//((index4 + u)/4) / k = index / k
//===================OLD fashion========================
//index4 / row_lengthv -> field_index
//x2.x = X2[(index4    ) / row_lengthv];
//x2.y = X2[(index4 + 1) / row_lengthv];
//x2.z = X2[(index4 + 2) / row_lengthv];
//x2.w = X2[(index4 + 3) / row_lengthv];
//===================OLD fashion========================

__global__ void quadratic_dual2D_field_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2, int row_lengthv,
	float k11, float k12, float k22,
	float k1, float k2,
	float C,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 y;//Y = k11*X1^2 + k12*X1*X2  + k22*X2^2 + k1*X1 + k2*X2 + C
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);
		float x2 = X2[index4 / row_lengthv];
		
		//Y = k11*X1^2 + k1*X1 + C = X1*(k11*X1 + k1)
		y.x  = x1.x * (k11 * x1.x + k1);
		y.y  = x1.y * (k11 * x1.y + k1);
		y.z  = x1.z * (k11 * x1.z + k1);
		y.w  = x1.w * (k11 * x1.w + k1);
		
		//Y = k11*X1^2 + k12*X1*X2  + k22*X2^2 + k1*X1 + k2*X2 + C
		float xv2 = x2 * (k22 * x2 + k2) + C;
		float kv12 = k12 * x2;

		y.x += xv2 + kv12 * x1.x;
		y.y += xv2 + kv12 * x1.y;
		y.z += xv2 + kv12 * x1.z;
		y.w += xv2 + kv12 * x1.w;

		within_width(y, index4, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __quadratic_dual2D_field(cudaStream_t stream,
	const float* X1,
	const float* X2, int row_lengthv,
	float k11, float k12, float k22,
	float k1, float k2, 
	float C,
	float* Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { quadratic_dual2d_field_k4_small(stream, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride); return; }
	quadratic_dual2d_field_k4(stream, 5, 2, X1, X2, row_lengthv, k11, k12, k22, k1, k2, C, Y, lengthv, width, stride);
}


#endif