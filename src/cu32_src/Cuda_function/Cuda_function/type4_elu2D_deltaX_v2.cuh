#pragma once

#ifndef ELU_2D_DELTAX_V2_H
#define ELU_2D_DELTAX_V2_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
//V2: holdX(), X is not changed
#ifndef ELU_2D_DELTAX_V2_CALL
#define ELU_2D_DELTAX_V2_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define elu2d_deltaX_v2_k4(stream, LB, LT, deltaX, deltaY, X, alpha, k, lengthv, width, stride)\
	elu2D_deltaX_v2_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(deltaX, deltaY, X, alpha, k, lengthv, width, stride)

#define elu2d_deltaX_v2_k4_small(stream,  deltaX, deltaY, X, alpha, k, lengthv, width, stride)\
	elu2D_deltaX_v2_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(deltaX, deltaY, X, alpha, k, lengthv, width, stride)

#endif


#ifndef ELU_2D_DELTAX_V2_KERNEL
#define ELU_2D_DELTAX_V2_KERNEL

//X >  0: Y = alpha * X          , Y' = alpha
//X <= 0: Y = alpha * k*(e^X - 1), Y' = alpha * k * e^X
//Y' = (X > 0) * alpha + (X <= 0)*(alpha * k*e^X)
//Y' = alpha * { (X > 0) + (1 - (X > 0)) * k*e^X }  
//Y'/alpha = (X > 0) + (1 - (X > 0)) * k*e^X
//Y'/alpha = (X > 0)*(1 - k*e^X) + k*e^X
//Y' = alpha *{ (X > 0)*(1 - k*e^X) + k*e^X }
//Step:
//<1> u =  k*e^X
//<2> y' = alpha *{ (X > 0)*(1 - u) + u }
//<3> dx = dy * y'

__global__ void elu2D_deltaX_v2_kernel_4(
	float* __restrict__ deltaX,
	const float* __restrict__ deltaY,
	const float* __restrict__ X,
	float alpha, float k,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	const float beta = alpha * k - alpha;
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dy = *(float4*)(deltaY + index4);
		float4 x = *(float4*)(X + index4);

		float4 u;//<1> u =  k*e^X
		u.x = k * expf(x.x);
		u.y = k * expf(x.y);
		u.z = k * expf(x.z);
		u.w = k * expf(x.w);

		float4 dx;//<2> y' = alpha *{ (X > 0)*(1 - u) + u }
		dx.x = alpha * ((x.x > 0) *(1.0f - u.x) + u.x);
		dx.y = alpha * ((x.y > 0) *(1.0f - u.y) + u.y);
		dx.z = alpha * ((x.z > 0) *(1.0f - u.z) + u.z);
		dx.w = alpha * ((x.w > 0) *(1.0f - u.w) + u.w);

		simdMul4(dx, dx, dy);//<3> deltaX = deltaY * y'

		within_width(dx, index4, stride, width);
		*(float4*)(deltaX + index4) = dx;
	}
}

#endif


void __elu2D_deltaX_v2(cudaStream_t stream,
	float* deltaX,
	const float* deltaY,
	const float* X, float alpha, float k,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { elu2d_deltaX_v2_k4_small(stream, deltaX, deltaY, X, alpha, k, lengthv, width, stride); return; }
	elu2d_deltaX_v2_k4(stream, 5, 2, deltaX, deltaY, X, alpha, k, lengthv, width, stride);
}

#endif