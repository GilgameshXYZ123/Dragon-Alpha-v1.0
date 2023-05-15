#pragma once

#ifndef ADD_DIV_2D_FIELD_H
#define ADD_DIV_2D_FIELD_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADD_DIV_2D_FIELD_CALL
#define ADD_DIV_2D_FIELD_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define add_div2d_field_k4(stream, LB, LT, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)\
	add_div2D_field_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)

#define add_div2d_field_k4_small(stream, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)\
	add_div2D_field_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)

#endif


#ifndef ADD_DIV_2D_FIELD_KERNEL
#define ADD_DIV_2D_FIELD_KERNEL

//X2.length = X3.length = X1.height = field_length = lengthv / row_lengthv
//Y, X1  belongs to Tensor[field_length, row_size]
//X2, X3 belongs to Tensor[field_length]
//for each field:
//	Y[i] = (alpha*X1[i] + beta*X2 + gamma) / (X3 + delta)

//deltaX1 = deltaY * d"Y" / d"X1"
//deltaX1 = deltaY * {alpha / (X3 + delta)}
//deltaX1 = (alpha * deltaY) / (X3 + delta)
//deltaX1 = div2D_field(alpha, deltaY, 0, X3 + delta)

__global__ void add_div2D_field_kernel_4(
	const float* __restrict__ X1,
	const float* __restrict__ X2,//[row_lengthv]
	const float* __restrict__ X3, int row_lengthv,//[row_lengthv]
	float alpha, float beta, float gamma, float delta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1 + index4);

		int row_index4 = index4 / row_lengthv;
		float x2 = X2[row_index4];
		float x3 = X3[row_index4];
		
		float xv2 = beta * x2 + gamma;
		float xv3 = 1.0f / (x3 + delta);

		float4 y;//Y[i] = (alpha*X1[i] + beta*X2 + gamma) / (X3 + delta)
		y.x = (alpha * x1.x + xv2) * xv3;
		y.y = (alpha * x1.y + xv2) * xv3;
		y.z = (alpha * x1.z + xv2) * xv3;
		y.w = (alpha * x1.w + xv2) * xv3;

		within_width_zero_nan(y, index4, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __add_div2D_field(cudaStream_t stream,
	const float* X1,
	const float* X2,
	const float* X3, int row_lengthv,
	float alpha, float beta, float gamma, float delta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { add_div2d_field_k4_small(stream, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride); return; }
	add_div2d_field_k4(stream, 5, 2, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride);
}

#endif