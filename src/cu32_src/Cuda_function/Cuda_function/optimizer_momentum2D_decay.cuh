#pragma once

#ifndef MOMENTUM_2D_DECAY_H
#define MOMENTUM_2D_DECAY_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef MOMENTUM_2D_DECAY_CALL
#define MOMENTUM_2D_DECAY_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define momentum2d_decay_k4(stream, LB, LT, W, V, a1, a2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	momentum2D_decay_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V,a1,a2, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

#define momentum2d_decay_k4_small(stream, W, V, a1, a2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	momentum2D_decay_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V,a1,a2, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

#endif


#ifndef MOMENTUM_2D_DECAY_KERNEL
#define MOMENTUM_2D_DECAY_KERNEL

__global__ void momentum2D_decay_kernel_4(
	const float* __restrict__ W,
	float* __restrict__ V, float a1, float a2,
	const float* __restrict__ deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		//load data------------------------------------------------
		float4 dw = *(float4*)(deltaW + index4);
		float4 v = *(float4*)(V + index4);
		float4 w = *(float4*)(W + index4);

		//compute result-------------------------------------------
		//L2: W^2 -> L2coef * W, L1: W   -> L1coef * sign(W)
		dw.x += L1coef * SIGN(w.x) + L2coef * w.x;
		dw.y += L1coef * SIGN(w.y) + L2coef * w.y;
		dw.z += L1coef * SIGN(w.z) + L2coef * w.z;
		dw.w += L1coef * SIGN(w.w) + L2coef * w.w;

		//V = a1*V + a2*deltaW  
		v.x = a1 * v.x + a2 * dw.x;
		v.y = a1 * v.y + a2 * dw.y;
		v.z = a1 * v.z + a2 * dw.z;
		v.w = a1 * v.w + a2 * dw.w;
		
		//W = W - lr_t * V       
		w.x -= v.x * lr_t;
		w.y -= v.y * lr_t;
		w.z -= v.z * lr_t;
		w.w -= v.w * lr_t;

		//write data-----------------------------------------------
		within_width(v, index4, stride, width);
		within_width(w, index4, stride, width);
		*(float4*)(V + index4) = v;//update velocity
		*(float4*)(W + index4) = w;//update weight
	}
}

#endif


void __momentum2D_decay(cudaStream_t stream,
	const float* W,
	float* V, float a1, float a2,
	const float* deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { momentum2d_decay_k4_small(stream, W, V, a1, a2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride); return; }
	momentum2d_decay_k4(stream, 5, 2, W, V, a1, a2, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride);
}

#endif