#pragma once

#ifndef ADAMW_2D_H
#define ADAMW_2D_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADAMW_2D_CALL
#define ADAMW_2D_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define adamW2d_k4(stream, LB, LT, W, V, a1, a2, S, b1, b2, eps_t, deltaW, lr_t, lr, L1coef, L2coef, lengthv, width, stride)\
	adamW2D_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, deltaW,lr_t,lr, L1coef, L2coef, lengthv,width,stride)

#define adamW2d_k4_small(stream, W, V, a1, a2, S, b1, b2, eps_t, deltaW, lr_t, lr, L1coef, L2coef, lengthv, width, stride)\
	adamW2D_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V,a1,a2, S,b1,b2,eps_t, deltaW,lr_t,lr, L1coef, L2coef, lengthv,width,stride)

#endif


#ifndef ADAMW_2D_KERNEL
#define ADAMW_2D_KERNEL

__global__ void adamW2D_kernel_4(
	const float* __restrict__ W,
	float* __restrict__ V, float a1, float a2,
	float* __restrict__ S, float b1, float b2, float eps_t,
	const float* __restrict__ deltaW,
	float lr_t, float lr,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//V = a1 * V + a2 * deltaW        
	//S = b1 * S + b2 * deltaW ^ 2
	//W = W - lr_t * V / (sqrt(S) + e) 
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dw = *(float4*)(deltaW + index4);
		float4 v  = *(float4*)(V      + index4);
		float4 s  = *(float4*)(S      + index4);
		float4 w  = *(float4*)(W      + index4);

		//V = a1*V + a2*deltaW  
		v.x = a1 * v.x + a2 * dw.x;
		v.y = a1 * v.y + a2 * dw.y;
		v.z = a1 * v.z + a2 * dw.z;
		v.w = a1 * v.w + a2 * dw.w;

		//S = b1*S + b2*deltaW^2
		s.x = b1 * s.x + b2 * (dw.x * dw.x);
		s.y = b1 * s.y + b2 * (dw.y * dw.y);
		s.z = b1 * s.z + b2 * (dw.z * dw.z);
		s.w = b1 * s.w + b2 * (dw.w * dw.w);
	
		float4 step;
		step.x = v.x / (sqrtf(s.x) + eps_t);
		step.y = v.y / (sqrtf(s.y) + eps_t);
		step.z = v.z / (sqrtf(s.z) + eps_t);
		step.w = v.w / (sqrtf(s.w) + eps_t);

		float4 regu;
		regu.x = L1coef * SIGN(w.x) + L2coef * w.x;
		regu.y = L1coef * SIGN(w.y) + L2coef * w.y;
		regu.z = L1coef * SIGN(w.z) + L2coef * w.z;
		regu.w = L1coef * SIGN(w.w) + L2coef * w.w;
		
		//adam with decoupled weight decay
		w.x -= lr_t * step.x + lr * regu.x;
		w.y -= lr_t * step.y + lr * regu.y;
		w.z -= lr_t * step.z + lr * regu.z;
		w.w -= lr_t * step.w + lr * regu.w;

		within_width(v, index4, stride, width);
		within_width(s, index4, stride, width);
		within_width(w, index4, stride, width);
		*(float4*)(V + index4) = v;//update the velocity
		*(float4*)(S + index4) = s;//update the standard deviation
		*(float4*)(W + index4) = w;//update weight
	}
}

#endif


void __adamW2D(cudaStream_t stream,
	const float* W,
	float* V, float a1, float a2,
	float* S, float b1, float b2, float eps,
	const float* deltaW, float lr_t, float lr,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { adamW2d_k4_small(stream, W, V, a1, a2, S, b1, b2, eps, deltaW, lr_t, lr, L1coef, L2coef, lengthv, width, stride); return; }
	adamW2d_k4(stream, 5, 2, W, V, a1, a2, S, b1, b2, eps, deltaW, lr_t, lr, L1coef, L2coef, lengthv, width, stride);
}

#endif