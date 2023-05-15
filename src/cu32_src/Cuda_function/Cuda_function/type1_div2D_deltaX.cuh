#pragma once

#ifndef DIV_2D_DELTAX_H
#define DIV_2D_DELTAX_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef DIV_2D_DELTAX_CALL
#define DIV_2D_DELTAX_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define div2d_deltaX_k4(stream, LB, LT, deltaX1, deltaX2, deltaY, X1, alpha1, beta1, X2, alpha2, beta2, lengthv, width, stride)\
	div2D_deltaX_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, alpha1, beta1, X2, alpha2, beta2, lengthv, width, stride)

#define div2d_deltaX_k4_small(stream, deltaX1, deltaX2, deltaY, X1, alpha11, beta1, X2, alpha2, beta2, lengthv, width, stride)\
	div2D_deltaX_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX1, deltaX2, deltaY, X1, alpha1, beta1, X2, alpha2, beta2, lengthv, width, stride)

#endif


#ifndef DIV_2D_DELTAX_KERNEL
#define DIV_2D_DELTAX_KERNEL

//Y = (a1*X1 + b1)/(a2*X2 + b2) + gamma
//dY / dX1 =  a1 / (a2*X2 + b2)
//dY / dX2 = -a2 * (a1*X1 + b1)/{(a2*X2 + b2)^2} 
//deltaX1 = (dY / dX1) * deltaY
//deltaX2 = (dY / dX2) * deltaY

__global__ void div2D_deltaX_kernel_4(
	float* __restrict__ deltaX1,
	float* __restrict__ deltaX2,
	const float* __restrict__ deltaY,
	const float* __restrict__ X1, float alpha1, float beta1,
	const float* __restrict__ X2, float alpha2, float beta2,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	float4 table[2]; table[0] = make_float4(0, 0, 0, 0);
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 x1 = *(float4*)(X1     + index4);
		float4 x2 = *(float4*)(X2     + index4);
		float4 dy = *(float4*)(deltaY + index4);

		simdLinear4(x1, alpha1, x1, beta1);//X1 -> (a1*X1 + b1)
		simdLinear4(x2, alpha2, x2, beta2);//X2 -> (a2*X2 + b2)

		float4 dx1;//d"Y"/d"X1"=  a1 / (a2*X2 + b2)
		dx1.x = alpha1 / x2.x;
		dx1.y = alpha1 / x2.y;
		dx1.z = alpha1 / x2.z;
		dx1.w = alpha1 / x2.w;

		float4 dx2;//d"Y"/d"X2" = -a2 * (a1*X1 + b1)/{(a2*X2 + b2)^2} 
		dx2.x = -alpha2 * x1.x / (x2.x * x2.x);
		dx2.y = -alpha2 * x1.y / (x2.y * x2.y);
		dx2.z = -alpha2 * x1.z / (x2.z * x2.z);
		dx2.w = -alpha2 * x1.w / (x2.w * x2.w);
		
		simdMul4(dx1, dx1, dy);//deltaX1 = (d"Y"/d"X1") * deltaY
		simdMul4(dx2, dx2, dy);//deltaX2 = (d"Y"/d"X2") * deltaY

		within_width_zero_nan(dx1, index4, table, stride, width);
		within_width_zero_nan(dx2, index4, table, stride, width);
		*(float4*)(deltaX1 + index4) = dx1;
		*(float4*)(deltaX2 + index4) = dx2;
	}
}

#endif


void __div2D_deltaX(cudaStream_t stream,
	float* deltaX1, float* deltaX2,
	const float* deltaY,
	const float *X1, float alpha1, float beta1,
	const float* X2, float alpha2, float beta2,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { div2d_deltaX_k4_small(stream, deltaX1, deltaX2, deltaY, X1, alpha1, beta1, X2, alpha2, beta2, lengthv, width, stride); return; }
	div2d_deltaX_k4(stream, 5, 2, deltaX1, deltaX2, deltaY, X1, alpha1, beta1, X2, alpha2, beta2, lengthv, width, stride);
}

#endif