#pragma once

#ifndef ELU_2D_DELTAX_V1_H
#define ELU_2D_DELTAX_V1_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V1: holdY(), Y is not changed
#ifndef ELU_2D_DELTAX_V1_CALL
#define ELU_2D_DELTAX_V1_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define elu2d_deltaX_v1_k4(stream, LB, LT, deltaX, deltaY, Y, alpha, k, lengthv, width, stride)\
	elu2D_deltaX_v1_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, k, lengthv, width, stride)

#define elu2d_deltaX_v1_k4_small(stream,  deltaX, deltaY, Y, alpha, k, lengthv, width, stride)\
	elu2D_deltaX_v1_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, Y, alpha, k, lengthv, width, stride)

#endif


#ifndef ELU_2D_DELTAX_V1_KERNEL
#define ELU_2D_DELTAX_V1_KERNEL

//Y >  0: Y = alpha * X, Y' = alpha
//Y <= 0: Y = alpha * k*(e^X - 1), Y' = alpha * k*e^X =  Y + alpha*k
//Y' = (Y > 0)*alpha + (Y <= 0)*(Y + alpha*k)
//	= alpha + (Y <= 0)*(Y + alpha*k - alpha)
//let: beta = alpha*k - alpha
//Y' = alpha + (Y<=0)*(Y + beta)

#define ELU_DERI(y, alpha, beta) (alpha + (y<=0)*(y + beta))

__global__ void elu2D_deltaX_v1_kernel_4(
	float* __restrict__ deltaX, 
	const float* __restrict__ deltaY,
	const float* __restrict__ Y,
	float alpha, float k,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const float beta = alpha * k - alpha;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 y = *(float4*)(Y + index4);

		float4 dx;//find derivative
		dx.x = ELU_DERI(y.x, alpha, beta);
		dx.y = ELU_DERI(y.y, alpha, beta);
		dx.z = ELU_DERI(y.z, alpha, beta);
		dx.w = ELU_DERI(y.w, alpha, beta);

		simdMul4(dx, dx, dy);//deltaX = deltaY * driY

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __elu2D_deltaX_v1(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* Y, float alpha, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { elu2d_deltaX_v1_k4_small(stream, deltaX, deltaY, Y, alpha, k, lengthv, width, stride); return; }
	elu2d_deltaX_v1_k4(stream, 5, 2, deltaX, deltaY, Y, alpha, k, lengthv, width, stride);
}

#endif