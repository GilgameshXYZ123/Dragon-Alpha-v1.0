#pragma once

#ifndef ADD_DIV_2D_ROW_H
#define ADD_DIV_2D_ROW_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADD_DIV_2D_ROW_CALL
#define ADD_DIV_2D_ROW_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define add_div2d_row_k4(stream, LB, LT, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)\
	add_div2D_row_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)

#define add_div2d_row_k4_small(stream, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)\
	add_div2D_row_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride)

#endif


#ifndef ADD_DIV_2D_ROW_KERNEL
#define ADD_DIV_2D_ROW_KERNEL

//X2.lengthv = X3.lengthv = X1.width = row_size
//Y, X1  belongs to Tensor[field_length, row_size]
//X2, X3 belongs to Tensor[row_size]
//for each row:
//	Y[i] = (alpha*X1[i] + beta*X2 + gamma) / (X3 + delta)

//deltaX1 = deltaY * d"Y" / d"X1"
//deltaX1 = deltaY * {alpha / (X3 + delta)}
//deltaX1 = (alpha * deltaY) / (X3 + delta)
//deltaX1 = div2D_row([alpha, deltaY, 0], 
//                   ([1.0f, X3, delta], 0)

__global__ void add_div2D_row_kernel_4(
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

		int field_index4 = index4 % row_lengthv;
		float4 x2 = *(float4*)(X2 + field_index4);
		float4 x3 = *(float4*)(X3 + field_index4);

		float4 y;//Y[i] = (alpha*X1[i] + beta*X2 + gamma) / (X3 + delta)
		y.x = (alpha * x1.x + beta * x2.x + gamma) / (x3.x + delta);
		y.y = (alpha * x1.y + beta * x2.y + gamma) / (x3.y + delta);
		y.z = (alpha * x1.z + beta * x2.z + gamma) / (x3.z + delta);
		y.w = (alpha * x1.w + beta * x2.w + gamma) / (x3.w + delta);

		within_width_zero_nan(y, index, table, stride, width);
		*(float4*)(Y + index4) = y;
	}
}

#endif


void __add_div2D_row(cudaStream_t stream,
	const float* X1,
	const float* X2,
	const float* X3, int row_lengthv,
	float alpha, float beta, float gamma, float delta,
	float* __restrict__ Y,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { add_div2d_row_k4_small(stream, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride); return; }
	add_div2d_row_k4(stream, 5, 2, X1, X2, X3, row_lengthv, alpha, beta, gamma, delta, Y, lengthv, width, stride);
}

#endif