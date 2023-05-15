#pragma once

#ifndef ADAMAX_2D_DECAY_H
#define ADAMAX_2D_DECAY_H

//lengthv = height * stride
//stride = (width + 3)/4*4
//[lenghtv, stride] % 4 == 0
#ifndef ADAMAX_2D_DECAY_CALL
#define ADAMAX_2D_DECAY_CALL

//LB = log2(BLOCK_SIZE)
//LT = log2(THREAD_ELEMENT_VISIT_SIZE)

#define adamax2d_decay_k4(stream, LB, LT, W, V, a1, a2, S, b1, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adamax2D_decay_kernel_4\
		<<< lengthv>>LB>>LT, (1<<LB), 0, stream >>>\
			(W, V,a1,a2, S,b1,eps, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

#define adamax2d_decay_k4_small(stream, W, V, a1, a2, S, b1, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride)\
	adamax2D_decay_kernel_4\
		<<< 1, ((lengthv + 3) >> 2), 0, stream >>>\
			(W, V,a1,a2, S,b1,eps, deltaW,lr_t, L1coef, L2coef, lengthv,width,stride)

#endif


#ifndef ADAMAX_2D_DECAY_KERNEL
#define ADAMAX_2D_DECAY_KERNE

__global__ void adamax2D_decay_kernel_4(
	const float* __restrict__ W,
	float* __restrict__ V, float a1, float a2,
	float* __restrict__ S, float b1, float eps,
	const float* __restrict__ deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	int step = gridDim.x*blockDim.x, step4 = step << 2;
	int index = blockIdx.x*blockDim.x + threadIdx.x;

	//V = a1 * V + a2 * deltaW        
	//for S: use infinite norm to repalce 2 norm: S = fmaxf(b1*S, |deltaW|)  
	//W = W - lr_t * V / (sqrt(S) + e) 
	for (int index4 = index << 2; index4 < lengthv; index4 += step4)
	{
		float4 dw = *(float4*)(&deltaW[index4]);
		float4 w = *(float4*)(&W[index4]);
		dw.x += L1coef * SIGN(w.x) + L2coef * w.x;
		dw.y += L1coef * SIGN(w.y) + L2coef * w.y;
		dw.z += L1coef * SIGN(w.z) + L2coef * w.z;
		dw.w += L1coef * SIGN(w.w) + L2coef * w.w;

		float4 v = *(float4*)(&V[index4]);//V = a1 * V + a2 * deltaW  
		v.x = a1 * v.x + a2 * dw.x;
		v.y = a1 * v.y + a2 * dw.y;
		v.z = a1 * v.z + a2 * dw.z;
		v.w = a1 * v.w + a2 * dw.w;
		within_width(v, index4, stride, width);
		*(float4*)(&V[index4]) = v;//update the velocity

		float4 s = *(float4*)(&S[index4]);//S = fmaxf(b1*S, |deltaW|)  
		s.x = fmaxf(b1*s.x, fabsf(dw.x));
		s.y = fmaxf(b1*s.y, fabsf(dw.y));
		s.z = fmaxf(b1*s.z, fabsf(dw.z));
		s.w = fmaxf(b1*s.w, fabsf(dw.w));
		within_width(s, index4, stride, width);
		*(float4*)(&S[index4]) = s;//update the standard deviation

		//W = W - lr_t * (V / S) + e)
		w.x -= lr_t * v.x / (s.x + eps);
		w.y -= lr_t * v.y / (s.y + eps);
		w.z -= lr_t * v.z / (s.z + eps);
		w.w -= lr_t * v.w / (s.w + eps);
		within_width(w, index4, stride, width);
		*(float4*)(&W[index4]) = w;//update weight
	}
}

#endif


void __adamax2D_decay(cudaStream_t stream,
	const float* W,
	float* V, float a1, float a2,
	float* S, float b1, float eps,
	const float* deltaW, float lr_t,
	float L1coef, float L2coef,
	int lengthv, int width, int stride)
{
	if (lengthv < 256) { adamax2d_decay_k4_small(stream, W, V, a1, a2, S, b1, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride); return; }
	adamax2d_decay_k4(stream, 5, 2, W, V, a1, a2, S, b1, eps, deltaW, lr_t, L1coef, L2coef, lengthv, width, stride);
}

#endif